/** Prefer an English voice for SpeechSynthesis fallback (avoid Turkish locale reading English oddly). */

export function applyEnglishSpeechVoice(utterance: SpeechSynthesisUtterance) {
  utterance.lang = "en-US";
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

  const pick = () => {
    const voices = window.speechSynthesis.getVoices();
    const prefer = (v: SpeechSynthesisVoice) =>
      v.lang.toLowerCase().startsWith("en") &&
      /samantha|aaron|google us|google english|microsoft|united states|us english|daniel|alex|karen|susan|fred|moira|tessa|veena|zira|aria|jenny|guy|jason|mark/i.test(
        v.name
      );
    const v =
      voices.find(prefer) ||
      voices.find((x) => x.lang.toLowerCase().startsWith("en-us")) ||
      voices.find((x) => x.lang.toLowerCase().startsWith("en-gb")) ||
      voices.find((x) => x.lang.toLowerCase().startsWith("en"));
    if (v) utterance.voice = v;
  };
  pick();
  window.speechSynthesis.onvoiceschanged = pick;
  utterance.rate = 0.95;
}
