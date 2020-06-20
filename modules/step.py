from modules.loss import ctc_loss, ctc_label_error_rate


def valid_step(inputs, labels, input_lengths, label_lengths, model, loss_tb, ler_tb):
    logits = model.model(inputs, training=False)
    loss = ctc_loss(labels, logits, input_lengths, label_lengths)
    loss_tb.update_state(loss)
    ler = ctc_label_error_rate(labels, logits, input_lengths, label_lengths)
    ler_tb.update_state(ler)
    return logits.numpy()
