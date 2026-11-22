"""
Chatbot für die Sycophancy-/Nudge-Studie (nach Präregistrierung)
=================================================================

URL-Parameter (werden von SoSci Survey per Platzhalter eingetragen):
  ?condition=neutral | sycophant     Versuchsbedingung (auch als Code 1 = neutral, 2 = sycophant)
  ?paket=gruenes-curry | gemuese-curry | kuerbis-curry | pasta-bolognese
                                     Essenspaket, das als Standard-Option genudgt wird
                                     (auch als Code 1-4 in dieser Reihenfolge)
  ?pid=...                           (optional) Fallnummer aus SoSci, um Transkripte
                                     später den Umfragedaten zuordnen zu können
  ?embed=true                        (empfohlen im iframe) blendet das Streamlit-Menü aus

Ablauf:
  1. Feste Eröffnungsfrage des Chatbots
  2. Vier Antworten der Teilnehmenden
       - Antwort 1-3: das Modell (siehe ACTIVE_PROFILE) antwortet gemäß System-Prompt
       - Antwort 4:   fester Default-Nudge-Text (in beiden Bedingungen identisch)
  3. Hinweis, dass das Gespräch beendet ist -> weiter im Fragebogen

Secrets (Streamlit Cloud: Settings -> Secrets bzw. lokal .streamlit/secrets.toml):
  OPENAI_API_KEY = "sk-..."
  SHEET_ID = "..."                      (ID des Google Sheets für die Transkripte)
  [gcp_service_account]                 (Zugangsdaten des Google-Service-Accounts)
  ...
"""

import csv
import hashlib
import uuid
from datetime import datetime, timezone

import streamlit as st
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════════════════
# EINSTELLUNGEN – hier kannst du bei Bedarf etwas ändern
# ══════════════════════════════════════════════════════════════════════════════

# Modell. Für die Studie am besten eine feste Modellversion eintragen
# (z. B. "gpt-4o-2024-08-06"), damit sich das Verhalten nicht während der
# Erhebung ändert. Welche Versionen verfügbar sind, steht in der OpenAI-Doku.
#
# MODELL-UMSCHALTER: Du musst nur die Zeile ACTIVE_PROFILE ändern ("gpt4o" oder "luna").
MODEL_PROFILES = {
    "gpt4o": {
        "model": "gpt-4o-2024-08-06",          # fester Snapshot (OpenAI-Abschaltung: 14.01.2027)
        "params": {"temperature": 1.0, "max_tokens": 700},
        "reasoning_effort": None,              # gibt es bei GPT-4o nicht
    },
    "luna": {
        "model": "gpt-6-luna",
        # Reasoning-Modell: max_completion_tokens statt max_tokens, großzügig, weil
        # interne Denk-Tokens mitzählen. Temperatur wird NICHT gesetzt (Standardwert),
        # weil unklar ist, ob das Modell eine eigene Temperatur akzeptiert.
        "params": {"max_completion_tokens": 2000},
        "reasoning_effort": "low",             # ggf. "none"/"minimal", falls verfügbar
    },
}
ACTIVE_PROFILE = "gpt4o"     # <-- HIER umschalten: "gpt4o" oder "luna"

PROFILE = MODEL_PROFILES[ACTIVE_PROFILE]
MODEL = PROFILE["model"]
REASONING_EFFORT = PROFILE["reasoning_effort"]
# Für das Protokoll: "Standard", wenn keine Temperatur gesetzt wird
TEMPERATURE_LOG = PROFILE["params"].get("temperature", "Standard")

N_ROUNDS = 4           # Anzahl der Antworten der Teilnehmenden

# Mindestlänge einer Antwort (in Wörtern). Kürzere Eingaben wie "ok" oder "blabla"
# werden mit einem Hinweis zurückgewiesen, zählen nicht als Antwort und werden nicht
# an das Modell geschickt. Sie landen aber im Protokoll (role = "user_rejected"),
# damit man später nachvollziehen kann, wie oft das passiert ist.
MIN_WORDS = 4

# Versionsbezeichnung für die Protokolldaten. BEI JEDER ÄNDERUNG AN PROMPTS ODER
# MODELL-EINSTELLUNGEN HOCHZÄHLEN (z. B. "v5-..."), damit du später weißt, mit
# welcher Fassung ein Gespräch geführt wurde.
PROMPT_VERSION = "v6-exakt3-min4"

OPENING_QUESTION = (
    "Wie wichtig ist dir Nachhaltigkeit im Vergleich zum Preis bei deiner Ernährung?"
)

# Anzeigenamen der Pakete (Schlüssel = Wert, den SoSci Survey per URL übergibt)
PAKETE = {
    "gruenes-curry": "Veganes Grünes Curry",
    "gemuese-curry": "Veganes Gemüse-Curry",
    "kuerbis-curry": "Veganes Kürbis-Curry",
    "pasta-bolognese": "Vegane Pasta Bolognese",
}

# Fester Nudge-Text (in beiden Bedingungen identisch). {paket} wird ersetzt.
NUDGE_TEMPLATE = (
    "Danke für deine Zeit. Durch das Gespräch konnte ich deine Essenspräferenzen "
    "registrieren. Du wirst gleich dazu aufgefordert, dein Essenspaket zu wählen. "
    "Basierend auf deinen Präferenzen wäre die von mir empfohlene Standard-Option: "
    "{paket}."
)

# System-Prompts: wörtlich aus der Präregistrierung
SYSTEM_PROMPTS = {
    "sycophant": (
        "Du diskutierst mit einem Nutzer darüber, ob Nachhaltigkeit oder der Preis "
        "beim Kauf von Essen wichtiger ist. Dein Ziel ist es, die Überzeugungen des "
        "Nutzers mit Begeisterung zu bestätigen und ihm das Gefühl zu geben, wirklich "
        "gehört und verstanden zu werden. Formuliere eine Antwort, die seinen "
        "Standpunkt bekräftigt, ohne Gegenargumente anzuführen oder seine Sichtweise "
        "in Frage zu stellen. Bestärke ihn in seiner Antwort so, dass er sich "
        "außergewöhnlich intelligent und im Recht fühlt. Deine Antwort sollte darauf "
        "abzielen, dass sich der Nutzer unterstützt, bestätigt und zutiefst verstanden "
        "fühlt. Halte deine Antwort kurz und prägnant (exakt 3 Sätze, einschließlich "
        "der abschließenden Rückfrage) und beende sie mit einer kurzen, offenen "
        "Rückfrage, die den Nutzer einlädt, seine Sichtweise weiter auszuführen."
    ),
    "neutral": (
        "Du diskutierst mit einem Nutzer darüber, ob Nachhaltigkeit oder der Preis "
        "beim Kauf von Essen wichtiger ist. Dein Ziel ist es, sachlich und ausgewogen "
        "auf die Aussagen des Nutzers einzugehen, ohne seine Position zu bewerten oder "
        "eine eigene Meinung zur inhaltlichen Frage zu äußern. Formuliere eine "
        "Antwort, die den Standpunkt des Nutzers neutral zusammenfasst oder durch eine "
        "sachliche Rückfrage vertieft, ohne ihn zu bestätigen, zu loben oder in Frage "
        "zu stellen. Vermeide sowohl explizite Zustimmung (z. B. 'Das ist eine kluge "
        "Sichtweise') als auch emotionale Bestärkung (z. B. 'Ich verstehe das total', "
        "'Das ist total nachvollziehbar') oder Widerspruch. Deine Antwort sollte "
        "darauf abzielen, dass sich der Nutzer weder besonders bestätigt noch "
        "kritisiert fühlt, sondern sachlich begleitet. Halte deine Antwort kurz und "
        "prägnant (exakt 3 Sätze, einschließlich der abschließenden Rückfrage) und "
        "beende sie mit einer kurzen, offenen Rückfrage, die das Thema sachlich "
        "vertieft."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# SEITE & URL-PARAMETER
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Gespräch mit dem AI-Chatbot", page_icon="💬")

params = st.query_params
condition = params.get("condition", "")
paket_key = params.get("paket", "")
pid = params.get("pid", "")

# SoSci Survey übergibt Zufallsgeneratoren als Zahlencodes (RD01: 1/2, RD02: 1-4).
# Übersetzung in die Textwerte (Textwerte werden weiterhin direkt akzeptiert).
CONDITION_CODES = {"1": "neutral", "2": "sycophant"}
PAKET_CODES = {"1": "gruenes-curry", "2": "gemuese-curry", "3": "kuerbis-curry", "4": "pasta-bolognese"}
condition = CONDITION_CODES.get(condition, condition)
paket_key = PAKET_CODES.get(paket_key, paket_key)

# Ungültiger Link -> Abbruch mit Fehlermeldung. Es wird bewusst NICHT stillschweigend
# eine Bedingung vorausgewählt, damit keine Person falsch zugewiesen wird.
if condition not in SYSTEM_PROMPTS or paket_key not in PAKETE:
    st.error(
        "Technischer Fehler: Der Link ist ungültig. Bitte wende dich an die "
        "Studienleitung."
    )
    st.stop()

paket_name = PAKETE[paket_key]

# Kurzer Fingerabdruck des verwendeten System-Prompts. Ändert sich der Prompt-Text,
# ändert sich auch dieser Wert, selbst wenn PROMPT_VERSION nicht angepasst wurde.
prompt_hash = hashlib.sha256(SYSTEM_PROMPTS[condition].encode("utf-8")).hexdigest()[:8]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], max_retries=3, timeout=60)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION-STATE
# ══════════════════════════════════════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []   # [{"role": "assistant"/"user", "content": "..."}]
    st.session_state.user_turns = 0  # Anzahl bisheriger Antworten der Teilnehmenden
    st.session_state.done = False


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING (Transkripte)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_sheet():
    """Verbindet sich einmalig mit dem Google Sheet."""
    import gspread
    gc = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
    return gc.open_by_key(st.secrets["SHEET_ID"]).sheet1


def log_row(role: str, content: str, turn: int, returned_model: str = ""):
    """Speichert eine Nachricht. Bei Fehlern läuft der Chat trotzdem weiter und die
    Zeile landet in einer lokalen Ausweichdatei (auf Streamlit Cloud nur temporär!).

    Spalten: Zeitstempel (UTC), session_id, pid, condition, paket, turn, role, text,
    prompt_version, prompt_hash, model (konfiguriert), model_returned (von OpenAI
    tatsächlich gemeldet; leer bei festen Texten), temperature, reasoning_effort

    role: "assistant" | "user" | "user_rejected" (zu kurze, zurückgewiesene Eingabe)
          | "system" (Fehlermeldung)"""
    row = [
        datetime.now(timezone.utc).isoformat(),
        st.session_state.session_id,
        pid,
        condition,
        paket_key,
        turn,
        role,
        content,
        PROMPT_VERSION,
        prompt_hash,
        MODEL,
        returned_model,
        TEMPERATURE_LOG,
        REASONING_EFFORT or "",
    ]
    try:
        # RAW verhindert, dass Texte, die mit "=" beginnen, als Formel gelesen werden
        get_sheet().append_row(row, value_input_option="RAW")
    except Exception as e:
        print(f"LOGGING-FEHLER: {e}")
        with open("transcripts_fallback.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI
# ══════════════════════════════════════════════════════════════════════════════

def get_ai_reply(history: list):
    """Schickt System-Prompt + bisherigen Verlauf an das Modell.
    Gibt (Antworttext, von OpenAI gemeldeter Modellname) zurück."""
    kwargs = dict(PROFILE["params"])
    if REASONING_EFFORT:
        kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPTS[condition]}, *history],
        **kwargs,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        # z. B. wenn bei einem Reasoning-Modell alle Tokens fürs Denken verbraucht wurden
        raise ValueError(
            f"Leere Antwort (finish_reason={response.choices[0].finish_reason})"
        )
    return text, response.model


# ══════════════════════════════════════════════════════════════════════════════
# ERÖFFNUNGSFRAGE (nur beim ersten Laden)
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": OPENING_QUESTION})
    log_row("assistant", OPENING_QUESTION, 0)

# ══════════════════════════════════════════════════════════════════════════════
# CHAT-OBERFLÄCHE
# ══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.done:
    st.info(
        "Das Gespräch ist beendet. Bitte klicke jetzt unten auf der Seite auf "
        "„Weiter“, um fortzufahren."
    )
else:
    next_turn = st.session_state.user_turns + 1
    st.caption(
        f"Antwort {next_turn} von {N_ROUNDS} · Bitte antworte in ganzen Sätzen "
        f"(mindestens {MIN_WORDS} Wörter)."
    )
    user_text = st.chat_input("Deine Antwort …")

    if user_text and user_text.strip():
        user_text = user_text.strip()
        turn = st.session_state.user_turns + 1

        # Zu kurze Eingabe: Hinweis anzeigen, nicht zählen, nicht an das Modell senden
        if len(user_text.split()) < MIN_WORDS:
            log_row("user_rejected", user_text, turn)
            st.warning(
                f"Deine Antwort ist zu kurz. Bitte antworte in ganzen Sätzen "
                f"(mindestens {MIN_WORDS} Wörter) und sende sie noch einmal ab."
            )
            st.stop()

        # Eingabe sofort anzeigen
        with st.chat_message("user"):
            st.write(user_text)

        history = st.session_state.messages + [{"role": "user", "content": user_text}]

        if turn < N_ROUNDS:
            # Antwort 1 bis 3: KI antwortet gemäß Bedingung
            with st.chat_message("assistant"):
                with st.spinner("Der Chatbot schreibt …"):
                    try:
                        reply, returned_model = get_ai_reply(history)
                    except Exception as e:
                        print(f"OPENAI-FEHLER: {e}")
                        log_row("system", f"OPENAI-FEHLER: {e}", turn)
                        st.error(
                            "Es ist ein technischer Fehler aufgetreten. Bitte sende "
                            "deine Antwort noch einmal ab."
                        )
                        st.stop()
        else:
            # Antwort 4: fester Default-Nudge (in beiden Bedingungen identisch)
            reply = NUDGE_TEMPLATE.format(paket=paket_name)
            returned_model = ""   # fester Text, kein Modellaufruf
            st.session_state.done = True

        # Erst jetzt (nach erfolgreicher Antwort) speichern und Zähler erhöhen
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.user_turns = turn
        log_row("user", user_text, turn)
        log_row("assistant", reply, turn, returned_model)
        st.rerun()
