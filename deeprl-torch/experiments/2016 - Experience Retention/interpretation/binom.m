N = 10000;
e = 200;
ex = 8;

epsinb = N/e
probsamp_e = ex/epsinb

p_c(1) = 1;
for i = 1:100 
    p(i) = factorial(epsinb)/(factorial(i-1)*factorial(epsinb-(i-1)))*probsamp_e^(i-1) * (1-probsamp_e)^(epsinb-(i-1));
    p_c(i+1) = p_c(i) - p(i)
    
end


