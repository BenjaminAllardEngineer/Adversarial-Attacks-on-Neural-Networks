def t_fgsm_attack(image, espilon, model, false_target):
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(output, false_target)

    model.zero_grad()
    loss.backward()

    # Collect datagrad
    image_grad = image.grad.data
    sign_data_grad = image_grad.sign()

    perturbed_image = image - epsilon * sign_data_grad
    
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image