# Load FC adversarial examples and test them on the CNN

cnn = torch.load(model_file)
with open('data/pickle/fc_adv_examples', 'rb') as file:
    fc_adv_examples = pickle.load(file)

transfer_attack(cnn, fc_adv_examples, False)