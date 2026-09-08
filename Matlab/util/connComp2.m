function [new] = connComp2(maskR2, nCompTen, fill)
%nCompTen = numero di componenti connessi da mantenere (partendo dai più grandi)

[L,num] = bwlabel(maskR2, 8);

if (nCompTen >= num)
    nCompTen = num;
end

count = zeros(num,1);
for c=1:num
    count(c) = numel(find(L==c));
end

maxT = max(count);

for t=1:num-nCompTen   %for
    %fprintf('%d\t',t);
    mint = min(count);
    
    if (mint == maxT)
        break;
    end
    
    clear i j
    
    Ic = find(count == mint);
    Ic = Ic(1);
    
    I = find(L==Ic);
    maskR2(I) = 0;
    
    count(Ic) = maxT+1000;
      
end %end for



[L,num] = bwlabel(~maskR2, 8);
count = zeros(num,1);
for (c=1:num)
    count(c) = numel(find(L==c));
end

maxT = max(count);
for (t=1:num-nCompTen-1)   %for
    mint = min(count);
    
    if (mint == maxT)
        break;
    end
    
    clear i j
    
    Ic = find(count == mint);
    Ic = Ic(1);
    
    I = find(L==Ic);
    maskR2(I) = 0;
    
    count(Ic) = maxT+1000;
    
     
end %end for


if (fill)
    maskR2 = imfill(maskR2,'holes');
end



new = maskR2;



