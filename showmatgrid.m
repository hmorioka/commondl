function showmatgrid(X,parm)
% showmatgrid
% 
% --- Input ------------------------------------------------------
% X             : data for plot [imageSize x imageSize x Ndata]
% parm struct
%   .crange     : (option) color range [[a, b] | 'common']
%                 If this is not specified, the color range is adjusted
%                 for each panel
%   .Nrow       : (option) number of row
%   .Ncolumn    : (option) number of column
%   .title      : (option) figure title
%
% Version 1.0, July 1 2015
% Author: Hiroshi Morioka
% License: Apache License, Version 2.0
%

if nargin < 2, parm = []; end
if isfield(parm,'crange'), crange = parm.crange; else crange = []; end
if isfield(parm,'Nrow'), Nrow = parm.Nrow; else Nrow = []; end
if isfield(parm,'Ncolumn'), Ncolumn = parm.Ncolumn; else Ncolumn = []; end
if isfield(parm,'title'), title = parm.title; else title = []; end

% Plot ----------------------------------------------------
% ---------------------------------------------------------
Ndata = size(X,3);

fig = figure; hold on;
figpos = get(fig,'Position');
set(fig,'Position',[100, figpos(2), figpos(3)*1.7, figpos(3)*1.5]);
gridLineWidth = 1;

if isempty(Ncolumn), Ncolumn = ceil(sqrt(Ndata)); end
if isempty(Nrow), Nrow = ceil(Ndata/Ncolumn); end

for dn = 1:Ndata

    subplotX = 1/Ncolumn*rem(dn-1,Ncolumn)+0.005;
    subplotY = 1/Nrow*(Nrow-floor((dn-1)/Ncolumn)-1)+0.005;
    subplotW = 1/Ncolumn*0.95;
    subplotH = 1/Nrow*0.95;
    subplot('Position',[subplotX,subplotY,subplotW,subplotH]);
    hold on; 
    
    Xn = X(:,:,dn);
    
    imagesc(Xn);
    set(gca,'YDir','rev');
    axis equal
    
    if isempty(crange)
        caxis([-max(abs(Xn(:))), max(abs(Xn(:)))]);
    elseif ~ischar(crange)
        caxis(crange);
    elseif ischar(crange) && strcmp(crange,'common')
        caxis([-max(abs(X(:))), max(abs(X(:)))]);
    end
    
    cmaphot = hot(64);
    cmap = [cmaphot(:,[3,2,1]);
            1 1 1;
            flipud(cmaphot)];
    colormap(cmap)

    xlim([0.5, size(Xn,2)+0.5]);
    ylim([0.5,size(Xn,1)+0.5]);

    xrange = get(gca,'YLim');
    yrange = get(gca,'YLim');
    xtick = [round(min(xrange)*10)/10:1:round(max(xrange)*10)/10];
    ytick = [round(min(yrange)*10)/10:1:round(max(yrange)*10)/10];

    xgrid_x = repmat(xtick,2,1);
    xgrid_y = repmat(yrange',1,size(xtick,2));
    ygrid_x = repmat(xrange',1,size(ytick,2));
    ygrid_y = repmat(ytick,2,1);

    gridColor = [1 1 1]*0.9;
    plot(xgrid_x, xgrid_y, 'color',gridColor, 'LineWidth',gridLineWidth);
    plot(ygrid_x, ygrid_y, 'color',gridColor, 'LineWidth',gridLineWidth);

    outGridColor = [1 1 1]*0.5;
    plot(xgrid_x(:,[1,end]), xgrid_y(:,[1,end]), 'color',outGridColor, 'LineWidth',gridLineWidth);
    plot(ygrid_x(:,[1,end]), ygrid_y(:,[1,end]), 'color',outGridColor, 'LineWidth',gridLineWidth);

    set(gca,'XTickLabel',[])
    set(gca,'YTickLabel',[])
    
end

if ~isempty(title)
    set(fig,'Name',title);
end

