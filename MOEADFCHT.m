function MOEADFCHT(Global)
% <algorithm> <M>
%------------------------------- Reference --------------------------------
% Dong Han, Wenli Du, Yaochu Jin, Wei Du, and Guo Yu. "A fuzzy constraint 
% handling technique for decomposition-based constrained multi-and many-
% objective optimization." Information Sciences 597 (2022): 318-340.
%--------------------------------------------------------------------------

    %% Parameter setting
    [delta,nr] = Global.ParameterSet(0.9,2);

    %% Generate the weight vectors
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    T = ceil(Global.N/10);

    %% Detect the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);

    %% Generate random population
    Population = Global.Initialization();
    Z = min(Population.objs,[],1);
    Conmin = min(overall_cv(Population.cons));
    
    A     = ArchiveUpdate(Population,Global.N);
    %% Optimization
    while Global.NotTermination(A)
        CV    = sum(max(Population.cons,0),2);
        fr    = length(find(CV<=0))/Global.N
        sigma_obj = 0.3;
        sigma_cv = fr*10;
        Q = [];
%         For each solution
        for i = 1 : Global.N
            % Choose the parents
            if rand < delta
                P = B(i,randperm(end));
            else
                P = randperm(Global.N);
            end

            % Generate an offspring
            if contains(class(Global.problem),'LIRCMOP') || contains(class(Global.problem),'DOC') || contains(class(Global.problem),'CF')
                Offspring = DE(Population(i),Population(P(1)),Population(P(2)));
            else
                Offspring = GAhalf(Population(P(1:2)));
            end

            % Update the ideal point
            Z = min(Z,Offspring.obj);
            Conmin = min(Conmin,overall_cv(Offspring.con));
            Zmax  = max([Population.objs;Offspring.obj],[],1);
            Conmax = max(overall_cv([Population.cons;Offspring.con]));

            % Update the solutions in P by Tchebycheff approach
            g_old = max(abs(Population(P).objs-repmat(Z,length(P),1))./W(P,:),[],2);
            g_new = max(repmat(abs(Offspring.obj-Z),length(P),1)./W(P,:),[],2);
            
            cv_old = overall_cv(Population(P).cons);
            cv_new = overall_cv(Offspring.con);
            if Conmax > Conmin
                cv_old(cv_old > 0) = (cv_old(cv_old > 0)-Conmin)/(Conmax-Conmin);
                cv_new(cv_new > 0) = (cv_new(cv_new > 0)-Conmin)/(Conmax-Conmin);
            end
            
            new_old = LG(g_old - g_new,[sigma_obj 0]).*max(LG(cv_old - repmat(cv_new,length(P),1),[sigma_cv 0]),0.0001);
            old_new = LG(g_new - g_old,[sigma_obj 0]).*max(LG(repmat(cv_new,length(P),1) - cv_old,[sigma_cv 0]),0.0001);
            Population(P(find(new_old>=old_new,nr))) = Offspring;
            if sum(max(Offspring.cons,0),2) == 0
                Q = [Q Offspring];
            end
        end
        if size(Q,2) > 0
           A = ArchiveUpdate([A Q],Global.N);
        end
    end
end

function result = overall_cv(cv)
	cv(cv <= 0) = 0;cv = abs(cv);
	result = sum(cv,2);
end

function y = LG(x, para)
    if (x==0)
        y = repmat(0.5,length(x),1);
    else
        y = (1+exp(-1*(x-para(2))./repmat(para(1),length(x),1))).^(-1);
    end
end