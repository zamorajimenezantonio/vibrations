function [f,mag_vect]=calculateBode(w0,wend,n,input,output,matlabcores);

    A = cell2mat(struct2cell(load(strcat('A','_',num2str(input),'_',num2str(output),'.mat'))));
    B = cell2mat(struct2cell(load(strcat('B','_',num2str(input),'_',num2str(output),'.mat'))));
    C = cell2mat(struct2cell(load(strcat('C','_',num2str(input),'_',num2str(output),'.mat'))));
    D=[[0]];
    
    maxNumCompThreads(matlabcores);

    sys1 = ss(A,B,C,D);
    %w=w0:winc:wend;
    n=2^n;
    w=linspace(w0,wend,n);
    [mag,phase,wout] = bode(sys1,w);
    mag_vect=mag(:);
    f=wout/(2*pi);
    %loglog(f,mag_vect);
    bode_data=[f,mag_vect];
    csvwrite(strcat('bode_data','_',num2str(input),'_',num2str(output),'.csv'),bode_data);
    
end

