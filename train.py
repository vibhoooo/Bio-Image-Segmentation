def get_val_metrics(mod, loader):
    mod.eval()
    n_total_steps = len(train_loader)
    los_vall = 0
    accuracy_vall = 0
    sensitivity_vall = 0
    specificity_vall = 0
    precision_vall = 0
    f1_vall = 0
    js_vall = 0
    dc_vall = 0
    print('Entering the loop')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
        output = model(data)
        loss = criterion(output, target)
        los_vall += loss.item()
        accuracy_vall += get_accuracy(output, target)
        sensitivity_vall += get_sensitivity(output, target)
        specificity_vall += get_specificity(output, target)
        precision_vall += get_precision(output, target)
        f1_vall += get_F1(output, target)
        js_vall += get_JS(output, target)
        dc_vall += get_DC(output, target)
    los_vall = los_vall / n_total_steps
    accuracy_vall = accuracy_vall / n_total_steps
    sensitivity_vall = sensitivity_vall / n_total_steps
    specificity_vall = specificity_vall / n_total_steps
    precision_vall = precision_vall / n_total_steps
    f1_vall = f1_vall / n_total_steps
    js_vall = js_vall / n_total_steps
    dc_vall = dc_vall / n_total_steps
    return los_vall, accuracy_vall, sensitivity_vall, specificity_vall, precision_vall, f1_vall, js_vall, dc_vall
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = Option()
# model = R2U_Net(input_channels=3, nclasses=1)
model = TransUNet(img_dim=128,
                  in_channels=3,
                  out_channels=128,
                  head_num=4,
                  mlp_dim=512,
                  block_num=8,
                  patch_dim=16,
                  class_num=1)
model.to(device)
# train_loader = get_val_loader()
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0.00001)
criterion = nn.BCEWithLogitsLoss().cuda()
train_loader = get_train_loader()
val_loader = get_val_loader()

los = 0
num_batches = 0
accuracy_train = 0
sensitivity_train = 0
specificity_train = 0
precision_train = 0
f1_train = 0
js_train = 0
dc_train = 0

los_val = 0
accuracy_train_val = 0
sensitivity_train_val = 0
specificity_train_val = 0
precision_train_val = 0
f1_train_val = 0
js_train_val = 0
dc_train_val = 0

avg_loss = []
avg_accuracy_train = []
avg_sensitivity_train = []
avg_specificity_train = []
avg_precision_train = []
avg_f1_train = []
avg_js_train = []
avg_dc_train = []
epoch_store = []

avg_loss_val = []
avg_accuracy_val = []
avg_sensitivity_val = []
avg_specificity_val = []
avg_precision_val = []
avg_f1_val = []
avg_js_val = []
avg_dc_val = []

n_total_steps = len(train_loader)
print('Entering the loop')
for epoch in range(0, opt.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # data = sample_batched['image']
        # target = sample_batched['mask']
        data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
        # print(f'Data max is: {data.max()}')
        # print(f'Target max is: {target.max()}')
        optimizer.zero_grad()
        output = model(data)
#         output = torch.sigmoid(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{opt.epochs}], Step [{batch_idx + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        los += loss.item()
        accuracy_train += get_accuracy(output, target)
        sensitivity_train += get_sensitivity(output, target)
        specificity_train += get_specificity(output, target)
        precision_train += get_precision(output, target)
        f1_train += get_F1(output, target)
        js_train += get_JS(output, target)
        dc_train += get_DC(output, target)
#         break
    scheduler.step()
    avg_loss.append(los / n_total_steps)
    avg_accuracy_train.append(accuracy_train / n_total_steps)
    avg_precision_train.append(precision_train / n_total_steps)
    avg_sensitivity_train.append(sensitivity_train / n_total_steps)
    avg_specificity_train.append(specificity_train / n_total_steps)
    avg_f1_train.append(f1_train / n_total_steps)
    avg_js_train.append(js_train / n_total_steps)
    avg_dc_train.append(dc_train / n_total_steps)
    epoch_store.append(epoch + 1)
    los = 0
    accuracy_train = 0
    sensitivity_train = 0
    specificity_train = 0
    precision_train = 0
    f1_train = 0
    js_train = 0
    dc_train = 0
   
    # For validation set:
    
    
#     break
datas = {'Average Training Accuracy': avg_accuracy_train,
         'Average Training Loss': avg_loss,
         'Average Training Precision': avg_precision_train,
         'Average Training Sensitivity': avg_precision_train,
         'Average Training Specificity': avg_specificity_train,
         'Average Training F1': avg_f1_train,
         'Average Training JS': avg_js_train,
         'Average Training Dice Coefficient': avg_dc_train,
         'Epochs': epoch_store
         }

df = pd.DataFrame(datas)
df.to_csv("Attn_UNET_MODEL_METRICS_train.csv")
if opt.save_model:
    torch.save(model.state_dict(), 'Attn_UNET_Model_2.pt')

print('TRAIN 1 program run complete')
print(f'Train datas: {datas}')

# For validation DATA
# los_val,accuracy_train_val ,sensitivity_train_val ,specificity_train_val ,precision_train_val ,f1_train_val ,js_train_val ,dc_train_val  = get_val_metrics(loader=val_loader,mod=model)
# avg_loss_val.append(los_val)
# avg_accuracy_val.append(accuracy_train_val)
# avg_sensitivity_val.append(sensitivity_train_val)
# avg_specificity_val.append(specificity_train_val)
# avg_precision_val.append(precision_train_val)
# avg_f1_val.append(f1_train_val)
# avg_js_val.append(js_train_val)
# avg_dc_val.append(dc_train_val)
# #For VALIDATION DATA
# datas_val = {'Average Val Accuracy': avg_accuracy_train,
#          'Average Val Loss': avg_loss,
#          'Average Val Precision': avg_precision_train,
#          'Average Val Sensitivity': avg_precision_train,
#          'Average Val Specificity': avg_specificity_train,
#          'Average Val F1': avg_f1_train,
#          'Average Val JS': avg_js_train,
#          'Average Val Dice Coefficient': avg_dc_train,
#          }

# df = pd.DataFrame(datas_val)
# df.to_csv("Attn_UNET_MODEL_METRICS_VAL.csv")
# print('VAL saving complete')
# print(f'VAL data:{datas_val}')