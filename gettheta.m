function mytheta = gettheta(theta,sizes,ind)
k = size(sizes,2)-2;
theta_ind=zeros(1,k+2);

for i = 1:k+1
    theta_ind(i+1) =  theta_ind(i)+(sizes(i)+1)*sizes(i+1);
end
beg = theta_ind(ind);
mytheta = reshape(theta((beg+1):(beg+sizes(ind+1)*(sizes(ind)+1))),sizes(ind+1),(sizes(ind)+1));
end
