local torch = require 'torch'
local data = torch.Tensor{
  {68, 24, 20},
  {74, 26, 21},
  {80, 32, 24},
}
sum = 0
for i=1,10000 do
	a = math.random()
	if a < 0.9 then
		sum = sum + 1
	end
end
print(sum/10000)