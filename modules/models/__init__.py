def create_model(name, checkpoint_state, hparams):
    if name == 'CTC':
        from .ctc import SpeechNetwork
        return SpeechNetwork(checkpoint_state, hparams)
    else:
        raise Exception('Unknown model: ' + name)
