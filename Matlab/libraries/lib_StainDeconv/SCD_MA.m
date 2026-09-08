function  [d, Ref_Vecs, H, E, Bg] = SCD_MA(I, plotta)


%clc, close all;

    %% Load image initialize variables 
    Level=5;Io=255; NumBands=20; 
    %addpath('FastICA_25');

    [h,w,~]=size(I);
    Statcount=1;StatMesur=zeros(3,Level*4);

   
    %% Reshape I into 3*(h*w)
     B = double(I(:,:,3));
     G = double(I(:,:,2));
     R = double(I(:,:,1));
     RGB = [reshape(R,1,h*w); reshape(G,1,h*w); reshape(B,1,h*w)];
     OD_M = -log((RGB+1)/255); 
     
     % optical density of images
     OD = -log((double(I)+1)/255); 
 
    %set fillters and apply DWT
    wname = 'db8';
    % initialize Approximation bands with OD channels
    A1=OD(:,:,1);
    A2=OD(:,:,2);
    A3=OD(:,:,3);
    Bands=cell(Level,4);
     
    for i=1:Level
            %% Generate Bands  
            [A1,H1,V1,D1] = dwt2(A1,wname);
            [A2,H2,V2,D2] = dwt2(A2,wname);
            [A3,H3,V3,D3] = dwt2(A3,wname);
            LL=[];LH=[];HL=[];HH=[];

            %% concatenate subbands based on colour channel 
            LL(:,:,1)=A1;LL(:,:,2)=A2;LL(:,:,3)=A3;
            LH(:,:,1)=H1;LH(:,:,2)=H2;LH(:,:,3)=H3;
            HL(:,:,1)=V1;HL(:,:,2)=V2;HL(:,:,3)=V3;
            HH(:,:,1)=D1;HH(:,:,2)=D2;HH(:,:,3)=D3;
            Bands{i,1}=LL;Bands{i,2}=LH;
            Bands{i,3}=HL;Bands{i,4}=HH;

            %% Show concatenated bands
             %ShowSubbands(LL,LH,HL,HH,i);

            %% Normalize bands to have zero mean and unit variance
            LL_1=(LL-mean(LL(:)))/std(LL(:));
            LH_1=(LH-mean(LH(:)))/std(LH(:));
            HL_1=(HL-mean(HL(:)))/std(HL(:));
            HH_1=(HH-mean(HH(:)))/std(HH(:));

            %% Compute Non-Gaussian Messures         
            NumEle  =numel(LL_1(:));
            %L1      =[norm(LL_1(:),1)/NumEle norm(LH_1(:),1)/NumEle norm(HL_1(:),1)/NumEle norm(HH_1(:),1)/NumEle];  
            Kor     =[abs(kurtosis(LL_1(:))-3) abs(kurtosis(LH_1(:))-3) abs(kurtosis(HL_1(:))-3) abs(kurtosis(HH_1(:))-3)]; 


            z=1;    
            for s=Statcount:(Statcount+3)  
                 StatMesur(1,s)=Kor(z);
                 StatMesur(2,s)=i;%level
                 StatMesur(3,s)=z;%band
                 z=z+1;
            end
            Statcount=Statcount+4;



    end

    % Sort Kourtosis matrix
    [~,d2] = sort(StatMesur(1,:),'descend');
     StatMesur=StatMesur(:,d2);

    %% Concatenate Subbands
    
        Coff=Bands{1,1};%Bands{level,band}
        [r,c,~]=size(Coff);
        B = double(Coff(:,:,3));
        G = double(Coff(:,:,2));
        R = double(Coff(:,:,1));
        Coff = [reshape(R,1,r*c); reshape(G,1,r*c); reshape(B,1,r*c)];
        FinalSignal=Coff; %need to be changed
        %FinalSignal=OD_M;
    for i=1:NumBands
%         if(StatMesur(2,i)~=1 && StatMesur(3,i)~=1)
            
        
        Coff=Bands{StatMesur(2,i),StatMesur(3,i)};%Bands{level,band}
        [r,c,~]=size(Coff);
        B = double(Coff(:,:,3));
        G = double(Coff(:,:,2));
        R = double(Coff(:,:,1));
        Coff = [reshape(R,1,r*c); reshape(G,1,r*c); reshape(B,1,r*c)];
        FinalSignal=horzcat(FinalSignal,Coff);
        
        
%         end
    end
    %% apply ICA

    [A,~] = fastica(FinalSignal,'numOfIC',3, 'verbose', 'off');%W{1}{1} is 3*3 matrix (each row for one source

    %% Compute OD and density image and stain matrix
    Ref_Vecs= abs(A);
    
    
    
    
    %% normalize stain vector
    for z = 1 : 3
        % normalise vector length     
        len=sqrt((Ref_Vecs(1,z)*Ref_Vecs(1,z)) + (Ref_Vecs(2,z)*Ref_Vecs(2,z))+ (Ref_Vecs(3,z)*Ref_Vecs(3,z)));
        if (len ~= 0.0)
            Ref_Vecs(1,z)= Ref_Vecs(1,z)/len;
            Ref_Vecs(2,z)= Ref_Vecs(2,z)/len;
            Ref_Vecs(3,z)= Ref_Vecs(3,z)/len;
        end
    end
    
    %% sort to start with H 
      Temp                = Ref_Vecs;
    [~,c]               = max(Temp(1,:));
    Ref_Vecs(:,1)       = Temp(:,c);
    Temp(:,c)           = [];
    
    [~,c]               = min(Temp(1,:));
    Ref_Vecs(:,2)       = Temp(:,c);
    Temp(:,c)           = [];
    Ref_Vecs(:,3)       = Temp;

    %% compute density matrix and show results
       d=(Ref_Vecs)\OD_M;
      %d(d<0)=0;
     %% Show results
      H = Io*exp(-Ref_Vecs(:,1)*d(1,:));
      H = reshape(H', h, w, 3);
      H = uint8(H);

      E = Io*exp(-Ref_Vecs(:,2)*d(2,:));
      E = reshape(E', h, w, 3);
      E = uint8(E);

      Bg = Io*exp(-Ref_Vecs(:,3)*d(3,:));
      Bg = reshape(Bg', h, w, 3);
      Bg = uint8(Bg);
      


      %% Show seperated stain for the sample image
      if 0
          figure,subplot(141);imshow(I);axis off, title('Source');
          set(gcf,'units','normalized','outerposition',[0 0 1 1]);
          
          subplot(142),imshow(H);axis off, title('H');
          set(gcf,'units','normalized','outerposition',[0 0 1 1]);
          
          subplot(143),imshow(E);axis off, title('E');
          set(gcf,'units','normalized','outerposition',[0 0 1 1]);
          
          subplot(144),imshow(Bg);axis off, title('Bg');
          set(gcf,'units','normalized','outerposition',[0 0 1 1]);
          
         mtit(fh, [filename '; segmentation'], 'Interpreter', 'none', 'fontsize', 14, 'color', [1 0 0], 'xoff', .0, 'yoff', .0);   
      end

    %
end




