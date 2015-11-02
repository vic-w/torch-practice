require 'image'
require 'torch'
require 'nn'
require 'qtwidget'
require 'optim'
require 'cutorch'
require 'cunn'

l=image.lena()

window=qtwidget.newwindow(512,512)
image.display({image=l,win=window})
 
model = nn.Sequential()
model:add(nn.MulConstant(1/1000))
layer1 = nn.Linear(2,300)
model:add(layer1)
model:add(nn.ReLU())
layer2 = nn.Linear(300,300)
model:add(layer2)
model:add(nn.ReLU())
layer3 = nn.Linear(300,300)
model:add(layer3)
model:add(nn.Sigmoid())
layer4 = nn.Linear(300,3)
model:add(layer4)
model = require('weight-init')(model, 'xavier')
criterion = nn.MSECriterion()

function load_model()
   model = torch.load('model.t7')
end
pcall(load_model)

model:cuda()
criterion:cuda()


inputs={}
for yy=1,512 do
 for xx=1,512 do
  table.insert(inputs,{xx,yy})
 end
end
inputs = torch.Tensor(inputs)
targets = l:reshape(3,512*512):t()
shuffle = torch.randperm(512*512)

shuffled_inputs = torch.Tensor(512*512,2)
shuffled_targets = torch.Tensor(512*512,3)

for i = 1,512*512 do
   shuffled_inputs[{{i},{}}] = inputs[{{shuffle[i]},{}}]
   shuffled_targets[{{i},{}}] = targets[{{shuffle[i]},{}}]
end

inputs = inputs:cuda()
targets = targets:cuda()


params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}

x, dl_dx = model:getParameters()
t = 0


local feval = function(x_new)
    -- reset data
    batch_size = 2048*8

   if x ~= x_new then x:copy(x_new) end
    dl_dx:zero()

    batch_inputs = shuffled_inputs[{{t+1, t+batch_size},{}}]:cuda()
    batch_targets = shuffled_targets[{{t+1, t+batch_size},{}}]:cuda()
    -- perform mini-batch gradient descent
    local output = model:forward(batch_inputs)
    local loss = criterion:forward(output, batch_targets)
    gradoutput = criterion:backward(output, batch_targets)
    model:backward(batch_inputs, gradoutput)
    t = t+batch_size
    if t >= 512*512 then t = 0 end
    return loss, dl_dx
end


for n=1,10000000 do

 _, fs = optim.adagrad(feval, x, params)

 collectgarbage()

 print(n, 'err', fs[1])
 if n%128 == 0 then
    output = model:forward(inputs):t():reshape(3,512,512)
    image.display({image=output:double(),win=window})
    image.save('output.png', output)
 end
 if n%1280 == 0 then
    torch.save('model.t7', model)
 end
end

