"""
RSNA Knee Abnormality Detection - Multilingual Tri-State NLP Extractor.
Extracts high-fidelity tri-state labels (+1.0 = positive, 0.0 = normal/negated, NaN = unmentioned)
across multilingual radiology reports in Spanish, Greek, Cyrillic, and English.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.metrics.auc_metrics import TARGET_COLUMNS


def normalize_text(text: str) -> str:
    """Normalizes text, strips diacritical marks/accents across Latin and Greek, and lowers."""
    if not isinstance(text, str):
        return ""
    # Strip diacritics
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower().replace("\n", " ").strip()
    return text


class MultilingualKneeNLPExtractor:
    """
    Multilingual Tri-State Rule & Semantic Extractor for Knee Radiology Reports.
    Handles Spanish (85%), Greek (7.3%), Cyrillic (5.0%), and English (~2%).
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. Global Negation / Normalcy Modifiers across languages
        self.negation_patterns = [
            # Standalone basic negations
            r"\b(?:sin|no|without|free\s+of|negative\s+for|non|χωρις|δεν|без|не)\b",
            # Spanish
            r"sin\s+(?:hallazgos|alteraciones|signos|evidencia|lesion|patologia|rotura|desgarro|derrame|fractura)",
            r"no\s+se\s+(?:observa|aprecia|identifica|evidencia|visualiza)",
            r"dentro\s+de\s+la\s+normalidad",
            r"(?:integro|integra|conservado|conservada|normal|homogeneo)",
            # English
            r"no\s+(?:evidence|sign|tear|fracture|effusion)",
            r"intact|unremarkable|normal|preserved",
            r"without\s+(?:acute|significant|abnormality)",
            # Greek (Ελληνικά)
            r"χωρις\s+(?:παθολογικα|ρηξη|ιδιαιτερα|ευρηματα|συλλογη|καταγμα)",
            r"χωρις\s+στοιχεια",
            r"εντος\s+του\s+φυσιολογικου",
            r"φυσιολογικ(?:ος|η|ο|ους|ες|α)",
            r"ακεραι(?:ος|η|ο|ους|ες|α)",
            r"δεν\s+(?:παρατηρειται|διαπιστωνεται|αναδεικνυεται|ελεγχεται)",
            # Cyrillic (Български / Русский)
            r"без\s+(?:данни\s+за|патологични|видими|скъсване|излив|фрактура)",
            r"интактен|интактна|интактно|интактни",
            r"в\s+норма|нормална|нормален|нормално|нормални",
            r"не\s+се\s+(?:вижда|установява|визуализира)",
        ]
        self.neg_regex = re.compile("|".join(self.negation_patterns), re.IGNORECASE)

        # 2. Target Specific Anatomical & Pathology Lexicons
        self.lexicons = {
            "ACL": {
                "anatomy": [
                    r"lca", r"ligamento\s+cruzado\s+anterior", r"anterior\s+cruciate",
                    r"προσθι(?:ου|ος|ο)\s+χιαστ(?:ου|ος|ο)", r"προσθιο\s+χιαστο", r"πχσ",
                    r"предна\s+кръстна\s+връзка", r"пкв",
                ],
                "positive": [
                    r"rotura", r"desgarro", r"avulsion", r"disrrupcion", r"interrupcion", r"lesion",
                    r"tear", r"rupture", r"sprain",
                    r"ρηξη", r"διαταση", r"τραυματισμ",
                    r"скъсване", r"руптура", r"увреда", r"лезия",
                ],
            },
            "MCL": {
                "anatomy": [
                    r"lcm", r"ligamento\s+colateral\s+medial", r"ligamento\s+colateral\s+interno", r"medial\s+collateral",
                    r"εσω\s+πλαγι(?:ου|ος|ο)", r"εσω\s+πλαγιο", r"επσ",
                    r"медиална\s+колатерална\s+връзка", r"мкв",
                ],
                "positive": [
                    r"rotura", r"desgarro", r"esguince", r"edema", r"engrosamiento", r"lesion",
                    r"tear", r"rupture", r"sprain", r"edema",
                    r"ρηξη", r"διαταση", r"οιδημα",
                    r"скъсване", r"руптура", r"разтежение", r"едем",
                ],
            },
            "Medial Meniscus": {
                "anatomy": [
                    r"menisco\s+medial", r"menisco\s+interno", r"medial\s+meniscus",
                    r"εσω\s+μηνισκ(?:ου|ος|ο)", r"εσω\s+μηνισκο",
                    r"медиален\s+менискус", r"медиалния\s+менискус",
                ],
                "positive": [
                    r"rotura", r"desgarro", r"fisura", r"degeneracion", r"extrusion", r"lesion",
                    r"tear", r"rupture", r"degeneration", r"frayed",
                    r"ρηξη", r"εκφυλισ", r"αλλοιωσ", r"βλαβη",
                    r"скъсване", r"дегенера", r"руптура", r"фисура",
                ],
            },
            "Lateral Meniscus": {
                "anatomy": [
                    r"menisco\s+lateral", r"menisco\s+externo", r"lateral\s+meniscus",
                    r"εξω\s+μηνισκ(?:ου|ος|ο)", r"εξω\s+μηνισκο",
                    r"латерален\s+менискус", r"латералния\s+менискус",
                ],
                "positive": [
                    r"rotura", r"desgarro", r"fisura", r"degeneracion", r"discoid", r"quiste",
                    r"tear", r"rupture", r"degeneration", r"discoid",
                    r"ρηξη", r"εκφυλισ", r"δισκοειδ", r"κυστη",
                    r"скъсване", r"дегенера", r"дискоиден", r"киста",
                ],
            },
            "Medial OA": {
                "anatomy": [
                    r"compartimento\s+medial", r"compartimento\s+interno", r"femoro[- ]?tibial\s+medial",
                    r"εσω\s+διαμερισμ", r"εσω\s+κνημομηριαι",
                    r"медиален\s+отдел", r"медиалния\s+кондил",
                ],
                "positive": [
                    r"artrosis", r"osteoartrosis", r"gonartrosis", r"condropatia", r"disminucion\s+de\s+espacio", r"osteofit",
                    r"osteoarthritis", r"arthrosis", r"chondromalacia", r"cartilage\s+loss", r"narrowing",
                    r"οστεοαρθριτιδ", r"αρθρωση", r"χονδροπαθει", r"στενωση", r"οστεοφυτ",
                    r"остеоартроз", r"артроза", r"хондромалация", r"стеснение", r"остеофит",
                ],
            },
            "Lateral OA": {
                "anatomy": [
                    r"compartimento\s+lateral", r"compartimento\s+externo", r"femoro[- ]?tibial\s+lateral",
                    r"εξω\s+διαμερισμ", r"εξω\s+κνημομηριαι",
                    r"латерален\s+отдел", r"латералния\s+кондил",
                ],
                "positive": [
                    r"artrosis", r"osteoartrosis", r"gonartrosis", r"condropatia", r"osteofit",
                    r"osteoarthritis", r"arthrosis", r"chondromalacia", r"narrowing",
                    r"οστεοαρθριτιδ", r"χονδροπαθει", r"οστεοφυτ",
                    r"остеоартроз", r"артроза", r"хондропатия",
                ],
            },
            "PF OA": {
                "anatomy": [
                    r"patelofemoral", r"femoropatelar", r"femoro[- ]?rotulian", r"rotula", r"patella",
                    r"επιγονατιδ", r"επιγονατιδομηριαι",
                    r"пателофеморална", r"патела",
                ],
                "positive": [
                    r"artrosis", r"condromalacia", r"condropatia", r"osteofit", r"desgaste",
                    r"osteoarthritis", r"chondromalacia", r"cartilage\s+loss",
                    r"οστεοαρθριτιδ", r"χονδρομαλακυνση", r"χονδροπαθει",
                    r"остеоартроз", r"хондромалация", r"хондропатия",
                ],
            },
            "Effusion": {
                "anatomy": [
                    r"derrame", r"liquido\s+intraarticular", r"liquido\s+articular", r"efusion", r"hidrartrosis",
                    r"effusion", r"joint\s+fluid",
                    r"συλλογη\s+υγρου", r"ενδαρθρικη\s+συλλογη", r"ενδοαρθρικο\s+υγρο", r"υγρο",
                    r"ставен\s+излив", r"излив", r"течност\s+в\s+ставата",
                ],
                "positive": [
                    r"derrame", r"moderado", r"abundante", r"leve", r"aumento", r"presencia",
                    r"effusion", r"fluid", r"present", r"small", r"moderate", r"large",
                    r"συλλογη", r"υγρο", r"παρατηρειται\s+συλλογη", r"ικανης\s+εκτασης", r"αυξημενο", r"παρουσια",
                    r"излив", r"наличие", r"умерен", r"значителен", r"изразен",
                ],
            },
            "Synovitis": {
                "anatomy": [
                    r"sinovial", r"sinovitis", r"plica", r"synovial", r"synovitis",
                    r"υμενικ", r"υμενιτιδ", r"синови", r"синовиит",
                ],
                "positive": [
                    r"sinovitis", r"engrosamiento", r"hipertrofia", r"reaccion\s+inflamatoria",
                    r"synovitis", r"thickening", r"hypertrophy",
                    r"υμενιτιδα", r"παχυνση", r"υπερτροφια",
                    r"синовиит", r"задебеляване", r"хипертрофия",
                ],
            },
            "Baker's": {
                "anatomy": [
                    r"baker", r"popliteo", r"hueco\s+popliteo", r"popliteal",
                    r"μπεικερ", r"μπεικερ", r"ιγνυακ", r"κυστη\s+baker",
                    r"бейкер", r"поплитеална", r"киста\s+на\s+бейкер",
                ],
                "positive": [
                    r"quiste", r"distension", r"coleccion",
                    r"cyst", r"fluid",
                    r"κυστη", r"συλλογη",
                    r"киста", r"образувание",
                ],
            },
            "Contusion": {
                "anatomy": [
                    r"edema\s+oseo", r"contusion", r"impactacion", r"trabecular",
                    r"bone\s+marrow\s+edema", r"bone\s+bruise", r"contusion",
                    r"οστικο\s+οιδημα", r"οιδημα\s+μυελου", r"θλαση",
                    r"костномозъчен\s+едем", r"костен\s+едем", r"контузия",
                ],
                "positive": [
                    r"edema", r"contusion", r"edema\s+subcondral",
                    r"edema", r"bruise",
                    r"οιδημα", r"θλαση",
                    r"едем", r"контузия",
                ],
            },
            "Fracture": {
                "anatomy": [
                    r"fractura", r"fisura\s+osea", r"hundimiento", r"fracture",
                    r"καταγμα", r"ρωγμη", r"фрактура", r"счупване",
                ],
                "positive": [
                    r"fractura", r"arrancamiento", r"trazo",
                    r"fracture", r"avulsion",
                    r"καταγμα", r"αποσπαστικ",
                    r"фрактура", r"счупване",
                ],
            },
        }

    def extract_from_report(self, text: str) -> Dict[str, float]:
        """
        Extracts Tri-State findings for all 12 targets:
          - 1.0 = Explicitly present / positive
          - 0.0 = Explicitly normal / negated
          - np.nan = Unmentioned in report
        """
        norm = normalize_text(text)
        if not norm or len(norm) < 5:
            return {c: np.nan for c in TARGET_COLUMNS}

        results = {}
        sentences = re.split(r"[\.\;\n\•\-\*]+", norm)

        for target in TARGET_COLUMNS:
            lex = self.lexicons.get(target)
            if not lex:
                results[target] = np.nan
                continue

            anat_regex = re.compile("|".join(lex["anatomy"]), re.IGNORECASE)
            pos_regex = re.compile("|".join(lex["positive"]), re.IGNORECASE)

            target_state = np.nan

            # Scan sentences for target mentions
            for s in sentences:
                s = s.strip()
                if not s:
                    continue

                if anat_regex.search(s):
                    # Check if negated/normal
                    if self.neg_regex.search(s) and not pos_regex.search(s):
                        target_state = 0.0
                    elif pos_regex.search(s):
                        # Verify it's not a negated positive (e.g., "no se observa rotura")
                        pos_match = pos_regex.search(s)
                        prefix = s[:pos_match.start()]
                        if self.neg_regex.search(prefix) or self.neg_regex.search(s):
                            # Check if negation is in close proximity to the match
                            if self.neg_regex.search(prefix):
                                target_state = 0.0
                            else:
                                target_state = 1.0
                                break
                        else:
                            target_state = 1.0
                            break  # Confirmed positive takes highest priority
                    else:
                        if np.isnan(target_state):
                            target_state = 0.0

            results[target] = target_state

        return results
