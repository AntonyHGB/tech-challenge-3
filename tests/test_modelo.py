"""Testes do modelo baseline de triagem."""

from triagem.modelo import classificar_laudo, treinar_modelo


def test_modelo_reconhece_as_tres_classes():
    modelo = treinar_modelo()
    assert set(modelo.classes_) == {"normal", "atencao", "urgente"}


def test_laudo_grave_e_classificado_como_urgente():
    modelo = treinar_modelo()
    texto = "Paciente com hemorragia intracraniana aguda e desvio de linha média"
    urgencia, confianca = classificar_laudo(modelo, texto)
    assert urgencia == "urgente"
    assert 0 < confianca <= 1


def test_laudo_sem_alteracoes_e_classificado_como_normal():
    modelo = treinar_modelo()
    texto = "Exame de sangue dentro dos padrões de normalidade"
    urgencia, _ = classificar_laudo(modelo, texto)
    assert urgencia == "normal"
