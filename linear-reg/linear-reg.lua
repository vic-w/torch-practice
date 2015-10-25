require 'torch'
require 'nn'
require 'gnuplot'

month = torch.range(1,10)
price = torch.Tensor{28993,29110,29436,30791,33384,36762,39900,39972,40230,40146}

model = nn.Linear(1, 1)
criterion = nn.MSECriterion()

month_train = month:reshape(10,1)
price_train = price:reshape(10,1)

for i=1,1000 do
   price_predict = model:forward(month_train)
   err = criterion:forward(price_predict, price_train)
   print(i, err)
   model:zeroGradParameters()
   gradient = criterion:backward(price_predict, price_train)
   model:backward(month_train, gradient)
   model:updateParameters(0.01)
end

month_predict = torch.range(1,12)
local price_predict = model:forward(month_predict:reshape(12,1))
print(price_predict)

gnuplot.pngfigure('plot.png')
gnuplot.plot({month, price}, {month_predict, price_predict})
gnuplot.plotflush()
