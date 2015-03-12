function out = houtk(X,theta,sizes)
%out = cell [a1 z2 a2 z3 a3 z4 a4 ...]
kk=length(sizes);
out = cell(2*kk-1,1);
out{1} = [ones(1,size(X,1));X'];
for i = 1:(kk-1)
    mytheta = gettheta(theta,sizes,i);
    out{2*i} = mytheta*out{2*i-1};
    %a_temp= sigmoid(out{2*i});
    a_temp = sigmf(out{2*i},[1 0]);
    out{2*i+1}=[ones(1,size(out{2*i},2)); a_temp];
end

out{end}=a_temp;
