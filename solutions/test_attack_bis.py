# Test attack on the simple model
def test_attack_bis( model, test_loader, epsilon, size_limit=5 ):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        
        # Modify the format of data so that the simple model can read it
        data = data.view(data.shape[0], -1)
        
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue
    
        # Call FGSM Attack or another attack
        perturbed_data = fgsm_attack(data, epsilon, model, target)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < size_limit):
                adv_ex = perturbed_data.data.cpu().numpy()
                org = data.data.cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, org) )
        else:
            if len(adv_examples) < size_limit:
                adv_ex = perturbed_data.data.cpu().numpy()
                org = data.data.cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, org) )

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples