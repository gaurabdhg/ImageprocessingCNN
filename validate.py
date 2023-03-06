def validate(model):
      # validation
    best_val_acc=[0,0]
    correct = 0
    val_loss=0
    total = 0
    counter=0
    model.eval()
    
    with torch.no_grad():
        for data in val_loader:
            counter+=1
            images,labels=data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss += loss_fn(outputs,labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total

    if 100 * correct / total >= bestmodel['accuracy']: 
      bestmodel={'epoch':  epoch,
        'model_state_dict': model.state_dict(),
        'loss':   val_loss,
        'accuracy':   val_acc
        }
    print(f'Validation accuracy: {100 * correct / total: .3f}%   ,'
              f'Validation loss:{val_loss/counter:.3f}')
    foo3.append(val_loss/counter)
    foo4.append(100 * correct / total)
    return
