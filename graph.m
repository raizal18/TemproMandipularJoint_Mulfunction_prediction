classdef graph
    
    %
    % * developer Er.Abbas Manthiri S
    % * Mail ID abbasmanthiribe@gmail.com
    % * Developed in 2015b
    %
    methods(Static)
        function plot(x,Y,algn,titlen,xl,yl,xval,xa,ya,t,len,s,sname)
            
            %Fixed xaxis values
            
            %%
            %Mk-markers
            mk={'*','+','o','d','x','v','s','p','h','^','>','>','.'};
            %Col color
            col={[0 0 0],[1 0 0],[0 1 0],[0 0 1],[0.5 0 0.5],[0.5 0 0],[255 20 147]/255,[75 0 130]/255,[47 79 79]/255};
            
            h1=figure('name',titlen,'numbertitle','off');
            set(h1,'Position',[100 100 750 516]);
            for i=1:length(Y(:,1))
                p(i)=plot(x,Y(i,:),'color',col{i});
                p(i).Marker = mk{i};
                p(i).LineWidth = 2;
                hold on
            end
            grid on
            xlim(xa)
            ylim(ya)
            set(gca,'xticklabel',xval)
            xlabel(xl)
            ylabel(yl)
            if t==1
                title(titlen)
            end
            if len==1
                legend(algn,'location','eastoutside')
            else
                legend(algn,'location','best')
            end
            if s
                saveas(h1,strcat(sname,'.jpg'))
            end
        end
        
        function bar(Y,algn,titlen,xl,yl,xval,xa,ya,t,len,s,sname)
            %Col color
            col={[0.4660 0.6740 0.1880],[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250],[0 0 0],[1 0 0],[0 1 0],[0 0 1],[0.5 0 0.5],[0.5 0 0],[255 20 147]/255,[75 0 130]/255,[47 79 79]/255};
            temp=zeros(1,size(Y,1));
            Y=[temp;Y';temp];
            
            %set figure window
            h1=figure('name',titlen,'numbertitle','off');
            set(h1,'Position',[100 100 750 516]);
            %grp coding
            p=bar(Y);
            for i=1:length(Y(1,:))
                p(i).FaceColor=col{i};
                hold on
            end
            xlabel(xl)
            ylabel(yl)
            xlim(xa)
            ylim(ya)
            set(gca,'xticklabel',xval)
            if t==1
                title(titlen)
            end
            if len==1
                legend(algn,'location','eastoutside')
            else
                legend(algn,'location','best')
            end
            if s
                saveas(h1,strcat(sname,'.jpg'))
            end
            clear Y p h1
        end
    end
end