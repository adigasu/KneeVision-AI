import numpy as np
import pytest
from src.nlp.multilingual_extractor import MultilingualKneeNLPExtractor


def test_multilingual_extractor_spanish():
    extractor = MultilingualKneeNLPExtractor()
    text = "Rotura completa del ligamento cruzado anterior. Menisco interno integro. Derrame articular leve."
    findings = extractor.extract_from_report(text)
    
    assert findings["ACL"] == 1.0
    assert findings["Medial Meniscus"] == 0.0
    assert findings["Effusion"] == 1.0
    assert np.isnan(findings["Fracture"])


def test_multilingual_extractor_greek():
    extractor = MultilingualKneeNLPExtractor()
    text = "Πλήρης ρήξη του πρόσθιου χιαστού συνδέσμου. Ενδαρθρική συλλογή υγρού. Χωρίς κάταγμα."
    findings = extractor.extract_from_report(text)
    
    assert findings["ACL"] == 1.0
    assert findings["Effusion"] == 1.0
    assert findings["Fracture"] == 0.0
    assert np.isnan(findings["Medial Meniscus"])


def test_multilingual_extractor_cyrillic():
    extractor = MultilingualKneeNLPExtractor()
    text = "МР данни за скъсване на задния рог на медиалния менискус. Ставен излив. Без данни за фрактура."
    findings = extractor.extract_from_report(text)
    
    assert findings["Medial Meniscus"] == 1.0
    assert findings["Effusion"] == 1.0
    assert findings["Fracture"] == 0.0
    assert np.isnan(findings["ACL"])
