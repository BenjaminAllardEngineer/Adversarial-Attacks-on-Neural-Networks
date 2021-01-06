# Generate adversarial examples for the FC model
# Save some of them in a file
filename = 'data/pickle/fc_adv_examples'

# Create and export 200 adversarial examples for later with epsilon=0.05
fc_model = torch.load(model_2_file)
acc, ex = test_attack_bis(fc_model, test_loader, 0.05, size_limit=200)

with open('data/pickle/fc_adv_examples', 'wb') as file:
    pickle.dump(ex, file)