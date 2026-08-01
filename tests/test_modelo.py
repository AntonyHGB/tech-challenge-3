"""Testes do modelo baseline de triagem."""

from triagem.modelo import avaliar_modelo, classificar_laudo

LAUDO_CARDIOVASCULAR = (
    "Acute myocardial infarction with ST segment elevation and cardiogenic shock "
    "requiring immediate coronary reperfusion therapy."
)


def test_modelo_reconhece_as_tres_classes(modelo):
    assert set(modelo.classes_) == {"normal", "atencao", "urgente"}


def test_classificacao_retorna_classe_e_confianca(modelo):
    urgencia, confianca = classificar_laudo(modelo, LAUDO_CARDIOVASCULAR)
    assert urgencia in {"normal", "atencao", "urgente"}
    assert 0 < confianca <= 1


def test_metricas_ficam_acima_do_piso_do_baseline(modelo):
    metricas = avaliar_modelo(modelo)
    assert metricas["acuracia"] > 0.55
    assert metricas["f1_macro"] > 0.55
