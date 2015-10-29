require 'image'
require 'torch'
require 'nn'
require 'qtwidget'
require 'optim'
--require 'cutorch'
--print(  cutorch.getDeviceProperties(cutorch.getDevice()) )
--require 'cunn'

l=image.lena()
--l=image.load('test.jpg')
--l=l[{{1},{},{}}]
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
--layer4 = nn.Linear(300,300)
--model:add(layer4)
--model:add(nn.ReLU())
layer5 = nn.Linear(300,3)
model:add(layer5)

model = require('weight-init')(model, 'xavier')

--model:cuda()

--model:add(nn.Sigmoid())
--model:add(nn.Linear(20,3))
criterion = nn.MSECriterion()
--criterion:cuda()
x, dl_dx = model:getParameters()

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}



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

--inputs = inputs:cuda()
--targets = targets:cuda()

t = 0

local feval = function(x_new)
    -- reset data
    batch_size = 2048*8

   if x ~= x_new then x:copy(x_new) end
    dl_dx:zero()

    batch_inputs = shuffled_inputs[{{t+1, t+batch_size},{}}]
    batch_targets = shuffled_targets[{{t+1, t+batch_size},{}}]
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

 _, fs = optim.adagrad(feval, x, sgd_params)

 collectgarbage()
 --print('layer1 w', layer1.weight)
 --print('layer1 b', layer1.bias)
 --print('layer2 w', layer2.weight)
 --print('layer2 b', layer2.bias)
 print(n, 'err', fs[1])
 if n%128 == 0 then
    output = model:forward(inputs):t():reshape(3,512,512)
    image.display({image=output,win=window})
    image.save('output.png', output)
 end
end

