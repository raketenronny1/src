% generate_exploratory_outlier_visualizations.m
%
% Function to generate all exploratory outlier analysis plots (Plots 1-8).
% Called by run_comprehensive_outlier_processing.m
%
% Date: 2025-05-18

function generate_exploratory_outlier_visualizations(X_train_full_flat, ...
                                                    y_numeric_full_flat, y_cat_full_flat, Patient_ID_full_flat, ...
                                                    wavenumbers_roi, score_all_spectra, explained_A3, coeff_A3, k_model_A3, ...
                                                    T2_values_all, Q_values_all, T2_threshold, Q_threshold, ...
                                                    flag_T2_all, flag_Q_all, is_T2_only_all, is_Q_only_all, ...
                                                    is_T2_and_Q_all, is_OR_outlier_all, is_normal_all, P_vis)
    
    fprintf('\n--- Starting Generation of Exploratory Outlier Visualizations ---\n');

    % Ensure the figures directory exists (passed in P_vis.figuresPath_OutlierExploration)
    if ~isfolder(P_vis.figuresPath_OutlierExploration), mkdir(P_vis.figuresPath_OutlierExploration); end

    % For convenience inside this function, you might unpack some P_vis parameters
    datePrefix = P_vis.datePrefix;
    figuresPath = P_vis.figuresPath_OutlierExploration; % Use the specific path for these plots
    
    % Plotting parameters from P_vis (shorten for local use if desired)
    colorWHO1 = P_vis.colorWHO1;
    colorWHO3 = P_vis.colorWHO3;
    colorT2OutlierFlag = P_vis.colorT2OutlierFlag;
    colorQOutlierFlag = P_vis.colorQOutlierFlag;
    colorBothOutlierFlag = P_vis.colorBothOutlierFlag;
    plotFontSize = P_vis.plotFontSize;
    plotXLabel = P_vis.plotXLabel;
    plotYLabelAbsorption = P_vis.plotYLabelAbsorption;
    plotXLim = P_vis.plotXLim;

    % Rename variables to match the plotting code from run_exploratory_outlier_analysis.m
    % This makes copying easier.
    X = X_train_full_flat;
    y_numeric = y_numeric_full_flat;
    y_cat = y_cat_full_flat; % If needed by any plot
    Patient_ID = Patient_ID_full_flat; % If needed by any plot
    score = score_all_spectra;
    explained = explained_A3; % PCA explained variance per component
    coeff = coeff_A3;         % PCA coefficients (loadings)
    k_model = k_model_A3;     % Number of PCs in the model

    T2_values = T2_values_all;
    Q_values = Q_values_all;
    % T2_threshold and Q_threshold are already correctly named

    % Flags (ensure these map correctly to your plotting logic)
    % is_normal, is_T2_only, is_Q_only, is_T2_and_Q, is_OR_outlier
    % are already passed in with the _all suffix, so use them directly or rename:
    % Example:
    is_normal_plot = is_normal_all;
    is_T2_only_plot = is_T2_only_all;
    is_Q_only_plot = is_Q_only_all;
    is_T2_and_Q_plot = is_T2_and_Q_all;
    is_OR_outlier_plot = is_OR_outlier_all;
    
    flag_T2_plot = flag_T2_all; % Direct mapping
    flag_Q_plot = flag_Q_all;   % Direct mapping


 

 % PLOT 1: Q-Statistic / T2 Statistic individual Plots (Revised)
    fig1 = figure('Name', 'Individual T2 and Q Statistics'); fig1.Position = [50,500,900,650];
    tl1 = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
    sgtitle(tl1,'Individual T2 and Q Statistic Distributions','FontWeight','Normal','FontSize',plotFontSize+1);
    ax1a=nexttile(tl1);hdl1a=[];leg1a={};
    hdl1a(end+1)=plot(ax1a,find(~flag_T2),T2_values(~flag_T2),'o','Color',[0.7 0.7 0.7],'MarkerSize',3);leg1a{end+1}='T2 <= Thresh'; hold(ax1a,'on');
    if any(flag_T2),hdl1a(end+1)=plot(ax1a,find(flag_T2),T2_values(flag_T2),'x','Color',colorT2OutlierFlag,'MarkerSize',5);leg1a{end+1}='T2 > Thresh';end
    h_t2_line=yline(ax1a,T2_threshold,'--','Color',colorT2OutlierFlag,'LineWidth',1.5);leg1a{end+1}=sprintf('T2 Thresh=%.2f',T2_threshold);hdl1a(end+1)=h_t2_line;
    hold(ax1a,'off');ylabel(ax1a,'T^2 Value');title(ax1a,sprintf('Hotelling T^2 (k_{model}=%d)',k_model),'FontWeight','normal');legend(ax1a,hdl1a,leg1a,'Location','northeast');grid(ax1a,'on');set(ax1a,'FontSize',plotFontSize-1);
    ax1b=nexttile(tl1);hdl1b=[];leg1b={};
    hdl1b(end+1)=plot(ax1b,find(~flag_Q),Q_values(~flag_Q),'o','Color',[0.7 0.7 0.7],'MarkerSize',3);leg1b{end+1}='Q <= Thresh';hold(ax1b,'on');
    if any(flag_Q),hdl1b(end+1)=plot(ax1b,find(flag_Q),Q_values(flag_Q),'x','Color',colorQOutlierFlag,'MarkerSize',5);leg1b{end+1}='Q > Thresh';end
    h_q_line=yline(ax1b,Q_threshold,'--','Color',colorQOutlierFlag,'LineWidth',1.5);leg1b{end+1}=sprintf('Q Thresh=%.2g',Q_threshold);hdl1b(end+1)=h_q_line;
    hold(ax1b,'off');xlabel(ax1b,'Spectrum Index');ylabel(ax1b,'Q-Statistic');title(ax1b,sprintf('Q-Statistic (SPE) (k_{model}=%d)',k_model),'FontWeight','normal');legend(ax1b,hdl1b,leg1b,'Location','northeast');grid(ax1b,'on');set(ax1b,'FontSize',plotFontSize-1);
    exportgraphics(fig1,fullfile(figuresDir,sprintf('%s_VisFunc_Plot1_T2_Q_Individual.tiff',datePrefix)),'Resolution',300);
    savefig(fig1,fullfile(figuresDir,sprintf('%s_VisFunc_Plot1_T2_Q_Individual.fig',datePrefix)));
    fprintf('Visualization Function: Plot 1 (Individual T2/Q) saved.\n'); close(fig1);

    % PLOT 2: T2 vs Q Plot with Thresholds and Outlier Categories (Revised)
    fig2=figure('Name','T2 vs Q with Categories & Labeled Thresholds');fig2.Position=[100 100 800 650];hold on;hdl2=[];leg2={};
    if any(is_normal & y_numeric==1),hdl2(end+1)=plot(T2_values(is_normal&y_numeric==1),Q_values(is_normal&y_numeric==1),'o','MarkerSize',4,'MarkerFaceColor',colorWHO1,'Color',colorWHO1);leg2{end+1}='WHO1(Normal)';end
    if any(is_normal & y_numeric==3),hdl2(end+1)=plot(T2_values(is_normal&y_numeric==3),Q_values(is_normal&y_numeric==3),'s','MarkerSize',4,'MarkerFaceColor',colorWHO3,'Color',colorWHO3);leg2{end+1}='WHO3(Normal)';end
    if any(is_T2_only),hdl2(end+1)=plot(T2_values(is_T2_only),Q_values(is_T2_only),'s','MarkerSize',5,'MarkerEdgeColor',colorT2OutlierFlag,'MarkerFaceColor',colorT2OutlierFlag*0.7);leg2{end+1}='T2-only';end
    if any(is_Q_only),hdl2(end+1)=plot(T2_values(is_Q_only),Q_values(is_Q_only),'d','MarkerSize',5,'MarkerEdgeColor',colorQOutlierFlag,'MarkerFaceColor',colorQOutlierFlag*0.7);leg2{end+1}='Q-only';end
    if any(is_T2_and_Q),hdl2(end+1)=plot(T2_values(is_T2_and_Q),Q_values(is_T2_and_Q),'*','MarkerSize',6,'Color',colorBothOutlierFlag);leg2{end+1}='T2&Q (Consensus)';end
    line([T2_threshold T2_threshold],get(gca,'YLim'),'Color',colorT2OutlierFlag,'LineStyle','--','LineWidth',1.2,'HandleVisibility','off');
    line(get(gca,'XLim'),[Q_threshold Q_threshold],'Color',colorQOutlierFlag,'LineStyle','--','LineWidth',1.2,'HandleVisibility','off');
    text(T2_threshold,mean(get(gca,'YLim'))*0.9,sprintf(' T2 Th=%.2f',T2_threshold),'Color',colorT2OutlierFlag,'VerticalAlignment','top','HorizontalAlignment','left','FontSize',plotFontSize-2,'BackgroundColor','w','EdgeColor','k','Margin',1);
    text(mean(get(gca,'XLim')),Q_threshold*1.05,sprintf(' Q Th=%.2g',Q_threshold),'Color',colorQOutlierFlag,'VerticalAlignment','bottom','HorizontalAlignment','center','FontSize',plotFontSize-2,'BackgroundColor','w','EdgeColor','k','Margin',1);
    hold off;xlabel('Hotelling T^2');ylabel('Q-Statistic (SPE)');title(sprintf('T^2 vs. Q (k_{model}=%d)',k_model),'FontWeight','normal','FontSize',plotFontSize);
    if ~isempty(hdl2),legend(hdl2,leg2,'Location','NorthEast','FontSize',plotFontSize-1);end;grid on;set(gca,'FontSize',plotFontSize-1);
    exportgraphics(fig2,fullfile(figuresDir,sprintf('%s_VisFunc_Plot2_T2vQ_CategoriesAndThresholds.tiff',datePrefix)),'Resolution',300);
    savefig(fig2,fullfile(figuresDir,sprintf('%s_VisFunc_Plot2_T2vQ_CategoriesAndThresholds.fig',datePrefix)));
    fprintf('Visualization Function: Plot 2 (T2vsQ Categories) saved.\n'); close(fig2);

    % PLOT 3: T2 vs Q Plot only WHO-1 and WHO-3 (Raw Distribution)
    fig3 = figure('Name', 'T2 vs Q Raw Distribution by WHO Grade'); fig3.Position = [150 150 750 600]; hold on;
    hdl3 = []; leg3 = {}; idx_w1_plot3 = (y_numeric == 1); idx_w3_plot3 = (y_numeric == 3); idx_unk_plot3 = ~(idx_w1_plot3 | idx_w3_plot3);
    if any(idx_w1_plot3),hdl3(end+1)=scatter(T2_values(idx_w1_plot3), Q_values(idx_w1_plot3), 15, colorWHO1, 'o', 'filled', 'MarkerFaceAlpha',0.4); leg3{end+1}='WHO-1'; end
    if any(idx_w3_plot3),hdl3(end+1)=scatter(T2_values(idx_w3_plot3), Q_values(idx_w3_plot3), 15, colorWHO3, 's', 'filled', 'MarkerFaceAlpha',0.4); leg3{end+1}='WHO-3'; end
    if any(idx_unk_plot3),hdl3(end+1)=scatter(T2_values(idx_unk_plot3), Q_values(idx_unk_plot3), 15, [0.7 0.7 0.7], '^', 'filled', 'MarkerFaceAlpha',0.3); leg3{end+1}='Other/Unknown Grade'; end
    hold off; xlabel('Hotelling T^2'); ylabel('Q-Statistic (SPE)'); title(sprintf('T^2 vs. Q Raw Data Distribution (k_{model}=%d)',k_model), 'FontWeight','normal','FontSize',plotFontSize);
    if ~isempty(hdl3), legend(hdl3,leg3,'Location','best','FontSize',plotFontSize-1); end; grid on;set(gca,'FontSize',plotFontSize-1);
    exportgraphics(fig3, fullfile(figuresDir, sprintf('%s_VisFunc_Plot3_T2vQ_WHOgradesOnly.tiff',datePrefix)),'Resolution',300);
    savefig(fig3, fullfile(figuresDir,sprintf('%s_VisFunc_Plot3_T2vQ_WHOgradesOnly.fig',datePrefix)));
    fprintf('Visualization Function: Plot 3 (T2vQ Raw Dist) saved.\n'); close(fig3);

    % PLOT 4a: PCA Scores by WHO Grade Only (2D and 3D in one figure)
    fig4a = figure('Name', 'PCA Scores by WHO Grade Only'); fig4a.Position = [50 50 900 450]; tl4a = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); sgtitle(tl4a,'PCA Scores (Colored by WHO Grade Only)','FontWeight','Normal','FontSize',plotFontSize);
    ax4a1=nexttile(tl4a);hold(ax4a1,'on');hdl4a1=[];leg4a1={}; if any(idx_w1_plot3),hdl4a1(end+1)=scatter(ax4a1,score(idx_w1_plot3,1),score(idx_w1_plot3,min(2,size(score,2))),15,colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4a1{end+1}='WHO-1';end; if any(idx_w3_plot3),hdl4a1(end+1)=scatter(ax4a1,score(idx_w3_plot3,1),score(idx_w3_plot3,min(2,size(score,2))),15,colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4a1{end+1}='WHO-3';end; if any(idx_unk_plot3),hdl4a1(end+1)=scatter(ax4a1,score(idx_unk_plot3,1),score(idx_unk_plot3,min(2,size(score,2))),15,[0.7 0.7 0.7],'^','filled','MarkerFaceAlpha',0.2);leg4a1{end+1}='Other';end; hold(ax4a1,'off');xlabel(ax4a1,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4a1,sprintf('PC2(%.1f%%)',explained(min(2,length(explained)))));title(ax4a1,'2D: PC1 vs PC2','FontWeight','normal','FontSize',plotFontSize-1);if ~isempty(hdl4a1),legend(ax4a1,hdl4a1,leg4a1,'Location','best','FontSize',plotFontSize-2);end;axis(ax4a1,'equal');grid(ax4a1,'on');set(ax4a1,'FontSize',plotFontSize-1);
    ax4a2=nexttile(tl4a);if size(score,2)>=3,hold(ax4a2,'on');hdl4a2=[];leg4a2={}; if any(idx_w1_plot3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_w1_plot3,1),score(idx_w1_plot3,2),score(idx_w1_plot3,3),15,colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4a2{end+1}='WHO-1';end; if any(idx_w3_plot3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_w3_plot3,1),score(idx_w3_plot3,2),score(idx_w3_plot3,3),15,colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4a2{end+1}='WHO-3';end; if any(idx_unk_plot3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_unk_plot3,1),score(idx_unk_plot3,2),score(idx_unk_plot3,3),15,[0.7 0.7 0.7],'^','filled','MarkerFaceAlpha',0.2);leg4a2{end+1}='Other';end; hold(ax4a2,'off');view(ax4a2,-30,20);xlabel(ax4a2,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4a2,sprintf('PC2(%.1f%%)',explained(2)));zlabel(ax4a2,sprintf('PC3(%.1f%%)',explained(3)));title(ax4a2,'3D: PC1-PC2-PC3','FontWeight','normal','FontSize',plotFontSize-1);if ~isempty(hdl4a2),legend(ax4a2,hdl4a2,leg4a2,'Location','best','FontSize',plotFontSize-2);end;grid(ax4a2,'on');axis(ax4a2,'tight');else,text(0.5,0.5,'<3 PCs','Parent',ax4a2,'HorizontalAlignment','center');title(ax4a2,'3D: PC1-PC2-PC3','FontWeight','normal','FontSize',plotFontSize-1);end;set(ax4a2,'FontSize',plotFontSize-1);
    exportgraphics(fig4a,fullfile(figuresDir,sprintf('%s_VisFunc_Plot4a_PCA_Scores_WHOonly.tiff',datePrefix)),'Resolution',300);
    savefig(fig4a,fullfile(figuresDir,sprintf('%s_VisFunc_Plot4a_PCA_Scores_WHOonly.fig',datePrefix)));
    fprintf('Visualization Function: Plot 4a (PCA Scores WHO only) saved.\n'); close(fig4a);

    % PLOT 4b: PCA Scores with Outlier Categories Marked (2D and 3D in one figure)
    fig4b = figure('Name','PCA Scores with Outlier Categories');fig4b.Position=[100 50 900 700];tl4b=tiledlayout(2,1,'TileSpacing','compact','Padding','compact');sgtitle(tl4b,'PCA Score Plots (Outlier Categories Marked)','FontWeight','Normal','FontSize',plotFontSize);
    ax4b1=nexttile(tl4b);hold(ax4b1,'on');hdl4b1=[];leg4b1={}; if any(is_normal&y_numeric==1),hdl4b1(end+1)=scatter(ax4b1,score(is_normal&y_numeric==1,1),score(is_normal&y_numeric==1,min(2,size(score,2))),15,colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4b1{end+1}='WHO1(Normal)';end; if any(is_normal&y_numeric==3),hdl4b1(end+1)=scatter(ax4b1,score(is_normal&y_numeric==3,1),score(is_normal&y_numeric==3,min(2,size(score,2))),15,colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4b1{end+1}='WHO3(Normal)';end; if any(is_T2_only),hdl4b1(end+1)=scatter(ax4b1,score(is_T2_only,1),score(is_T2_only,min(2,size(score,2))),25,colorT2OutlierFlag,'s','MarkerFaceColor',colorT2OutlierFlag*0.7);leg4b1{end+1}='T2-only';end; if any(is_Q_only),hdl4b1(end+1)=scatter(ax4b1,score(is_Q_only,1),score(is_Q_only,min(2,size(score,2))),25,colorQOutlierFlag,'d','MarkerFaceColor',colorQOutlierFlag*0.7);leg4b1{end+1}='Q-only';end; if any(is_T2_and_Q),hdl4b1(end+1)=scatter(ax4b1,score(is_T2_and_Q,1),score(is_T2_and_Q,min(2,size(score,2))),30,colorBothOutlierFlag,'*');leg4b1{end+1}='T2&Q (Consensus)';end; hold(ax4b1,'off');xlabel(ax4b1,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4b1,sprintf('PC2(%.1f%%)',explained(min(2,length(explained)))));title(ax4b1,'2D: PC1 vs PC2','FontWeight','normal','FontSize',plotFontSize-1);if ~isempty(hdl4b1),legend(ax4b1,hdl4b1,leg4b1,'Location','best','FontSize',plotFontSize-2);end;axis(ax4b1,'equal');grid(ax4b1,'on');set(ax4b1,'FontSize',plotFontSize-1);
    ax4b2=nexttile(tl4b);if size(score,2)>=3,hold(ax4b2,'on');hdl4b2=[];leg4b2={}; if any(is_normal&y_numeric==1),hdl4b2(end+1)=scatter3(ax4b2,score(is_normal&y_numeric==1,1),score(is_normal&y_numeric==1,2),score(is_normal&y_numeric==1,3),15,colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4b2{end+1}='WHO1(Normal)';end; if any(is_normal&y_numeric==3),hdl4b2(end+1)=scatter3(ax4b2,score(is_normal&y_numeric==3,1),score(is_normal&y_numeric==3,2),score(is_normal&y_numeric==3,3),15,colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4b2{end+1}='WHO3(Normal)';end; if any(is_T2_only),hdl4b2(end+1)=scatter3(ax4b2,score(is_T2_only,1),score(is_T2_only,2),score(is_T2_only,3),25,colorT2OutlierFlag,'s','MarkerFaceColor',colorT2OutlierFlag*0.7);leg4b2{end+1}='T2-only';end; if any(is_Q_only),hdl4b2(end+1)=scatter3(ax4b2,score(is_Q_only,1),score(is_Q_only,2),score(is_Q_only,3),25,colorQOutlierFlag,'d','MarkerFaceColor',colorQOutlierFlag*0.7);leg4b2{end+1}='Q-only';end; if any(is_T2_and_Q),hdl4b2(end+1)=scatter3(ax4b2,score(is_T2_and_Q,1),score(is_T2_and_Q,2),score(is_T2_and_Q,3),30,colorBothOutlierFlag,'*');leg4b2{end+1}='T2&Q (Consensus)';end; hold(ax4b2,'off');view(ax4b2,-30,20);xlabel(ax4b2,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4b2,sprintf('PC2(%.1f%%)',explained(2)));zlabel(ax4b2,sprintf('PC3(%.1f%%)',explained(3)));title(ax4b2,'3D: PC1-PC2-PC3','FontWeight','normal','FontSize',plotFontSize-1);if ~isempty(hdl4b2),legend(ax4b2,hdl4b2,leg4b2,'Location','best','FontSize',plotFontSize-2);end;grid(ax4b2,'on');axis(ax4b2,'tight');else,text(0.5,0.5,'<3 PCs','Parent',ax4b2,'HorizontalAlignment','center');title(ax4b2,'3D: PC1-PC2-PC3','FontWeight','normal','FontSize',plotFontSize-1);end;set(ax4b2,'FontSize',plotFontSize-1);
    exportgraphics(fig4b,fullfile(figuresDir,sprintf('%s_VisFunc_Plot4b_PCA_Scores_OutlierCats.tiff',datePrefix)),'Resolution',300);
    savefig(fig4b,fullfile(figuresDir,sprintf('%s_VisFunc_Plot4b_PCA_Scores_OutlierCats.fig',datePrefix)));
    fprintf('Visualization Function: Plot 4b (PCA Scores Outlier Cats) saved.\n'); close(fig4b);

    % PLOT 5: All PC Loadings (PC1 to k_model)
    num_pcs_for_loadings = k_model;
    if num_pcs_for_loadings > 0 && ~isempty(coeff)
        if num_pcs_for_loadings <= 3, ncols_loadings = 1; nrows_loadings = num_pcs_for_loadings;
        elseif num_pcs_for_loadings <= 8, ncols_loadings = 2; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings);
        else, ncols_loadings = 3; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings); end
        fig5 = figure('Name',sprintf('PCA Loadings (PC1-PC%d of T2/Q Model)',num_pcs_for_loadings));fig5.Position = [100 50 min(ncols_loadings*450,1350) min(nrows_loadings*220,880)];
        tl5 = tiledlayout(nrows_loadings,ncols_loadings,'TileSpacing','compact','Padding','tight'); sgtitle(tl5,sprintf('PCA Loadings for %d PCs used in T2/Q Model',num_pcs_for_loadings),'FontWeight','Normal','FontSize',plotFontSize);
        for pc_idx = 1:num_pcs_for_loadings
            if pc_idx > size(coeff,2), break; end; ax_l = nexttile(tl5); plot(ax_l,wavenumbers_roi,coeff(:,pc_idx),'LineWidth',1); title(ax_l,sprintf('PC%d Loadings (Expl.Var: %.2f%%)',pc_idx,explained(pc_idx)),'FontWeight','normal','FontSize',plotFontSize-1); ylabel(ax_l,'Loading Value','FontSize',plotFontSize-2);grid(ax_l,'on');set(ax_l,'XDir','reverse','XLim',plotXLim,'FontSize',plotFontSize-2);
            current_tile_info=get(ax_l,'Layout');current_col=mod(current_tile_info.Tile-1,ncols_loadings)+1;current_row=ceil(current_tile_info.Tile/ncols_loadings);
            if current_col~=1,set(ax_l,'YTickLabel',[]);end;if current_row~=nrows_loadings,set(ax_l,'XTickLabel',[]);end
        end; xlabel(tl5,plotXLabel,'FontSize',plotFontSize-1);
        exportgraphics(fig5,fullfile(figuresDir,sprintf('%s_VisFunc_Plot5_PCA_Loadings_kModel.tiff',datePrefix)),'Resolution',300);
        savefig(fig5,fullfile(figuresDir,sprintf('%s_VisFunc_Plot5_PCA_Loadings_kModel.fig',datePrefix)));
        fprintf('Visualization Function: Plot 5 (PCA Loadings) saved.\n'); close(fig5);
    else, fprintf('Skipping Plot 5 (Loadings) as k_model is 0 or coeffs empty.\n'); end

    % PLOT 6: Tiled Layout of Spectra for Distinct Outlier Categories
    fig6 = figure('Name', 'Spectra of Distinct Outlier Categories'); fig6.Position = [120 120 700 850];
    tl6 = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
    sgtitle(tl6,'Spectra by Distinct Outlier Category','FontWeight','Normal', 'FontSize', plotFontSize+1);
    outlier_cats_plot6 = {{'Q-only Flagged',is_Q_only,colorQOutlierFlag},{'T2-only Flagged',is_T2_only,colorT2OutlierFlag},{'T2&Q Flagged (Consensus)',is_T2_and_Q,colorBothOutlierFlag}};
    for cat_idx=1:3, ax_cat=nexttile(tl6);hold(ax_cat,'on');cat_title_base=outlier_cats_plot6{cat_idx}{1};cat_flag=outlier_cats_plot6{cat_idx}{2};cat_color_lines=outlier_cats_plot6{cat_idx}{3};
        spectra_cat=X(cat_flag,:);num_cat=sum(cat_flag);hdl_cat_mean=[];
        if num_cat>0
            plot(ax_cat,wavenumbers_roi,spectra_cat','Color',[cat_color_lines,0.1],'LineWidth',0.5,'HandleVisibility','off');
            mean_spec=mean(spectra_cat,1,'omitnan');
            if any(~isnan(mean_spec)),hdl_cat_mean=plot(ax_cat,wavenumbers_roi,mean_spec,'Color','k','LineWidth',1.5,'DisplayName',sprintf('Mean (n=%d)',num_cat)); uistack(hdl_cat_mean,'top'); end
        else, text(0.5,0.5,'No spectra','Parent',ax_cat,'HorizontalAlignment','center','FontSize',plotFontSize-1);end
        hold(ax_cat,'off');title(ax_cat,sprintf('%s (n=%d)',cat_title_base,num_cat),'FontWeight','normal','FontSize',plotFontSize-1);ylabel(ax_cat,plotYLabelAbsorption,'FontSize',plotFontSize-1);set(ax_cat,'XDir','reverse','XLim',plotXLim,'FontSize',plotFontSize-1);grid(ax_cat,'on');
        if cat_idx<3,set(ax_cat,'XTickLabel',[]);else,xlabel(ax_cat,plotXLabel,'FontSize',plotFontSize-1);end
        if ~isempty(hdl_cat_mean) && isgraphics(hdl_cat_mean),legend(ax_cat,hdl_cat_mean,'Location','northeast','FontSize',plotFontSize-2);end
    end
    drawnow; pause(0.1); % Ensure rendering
    exportgraphics(fig6,fullfile(figuresDir, sprintf('%s_VisFunc_Plot6_OutlierCategory_Spectra.tiff', datePrefix)),'Resolution',300);
    savefig(fig6,fullfile(figuresDir, sprintf('%s_VisFunc_Plot6_OutlierCategory_Spectra.fig', datePrefix)));
    fprintf('Visualization Function: Plot 6 (Outlier Category Spectra) saved.\n'); close(fig6);

  
% PLOT 6: Tiled Layout of Spectra for Distinct Outlier Categories (Revised Mean Line)
fig6 = figure('Name', 'Spectra of Distinct Outlier Categories'); fig6.Position = [120 120 700 850];
tl6 = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
sgtitle(tl6,'Spectra by Distinct Outlier Category','FontWeight','Normal', 'FontSize', P.plotFontSize+1);
outlier_cats_plot6 = {{'Q-only Flagged',is_Q_only,P.colorQOutlierFlag},{'T2-only Flagged',is_T2_only,P.colorT2OutlierFlag},{'T2&Q Flagged (Consensus)',is_T2_and_Q,P.colorBothOutlierFlag}};
for cat_idx=1:3, ax_cat=nexttile(tl6);hold(ax_cat,'on');cat_title_base=outlier_cats_plot6{cat_idx}{1};cat_flag=outlier_cats_plot6{cat_idx}{2};cat_color_lines=outlier_cats_plot6{cat_idx}{3};
    spectra_cat=X(cat_flag,:);num_cat=sum(cat_flag);hdl_cat_mean=[];
    if num_cat>0
        plot(ax_cat,wavenumbers_roi,spectra_cat','Color',[cat_color_lines,0.1],'LineWidth',0.5,'HandleVisibility','off');
        mean_spec=mean(spectra_cat,1,'omitnan');
        if any(~isnan(mean_spec)),hdl_cat_mean=plot(ax_cat,wavenumbers_roi,mean_spec,'Color','k','LineWidth',1.5,'DisplayName',sprintf('Mean (n=%d)',num_cat)); uistack(hdl_cat_mean,'top'); end
    else, text(0.5,0.5,'No spectra','Parent',ax_cat,'HorizontalAlignment','center');end
    hold(ax_cat,'off');title(ax_cat,sprintf('%s (n=%d)',cat_title_base,num_cat),'FontWeight','normal');ylabel(ax_cat,P.plotYLabelAbsorption);set(ax_cat,'XDir','reverse','XLim',P.plotXLim,'FontSize',P.plotFontSize-1);grid(ax_cat,'on');
    if cat_idx<3,set(ax_cat,'XTickLabel',[]);else,xlabel(ax_cat,P.plotXLabel);end
    if ~isempty(hdl_cat_mean) && isgraphics(hdl_cat_mean),legend(ax_cat,hdl_cat_mean,'Location','northeast','FontSize',P.plotFontSize-2);end
end
drawnow; pause(0.2);
tiffFile6 = fullfile(figuresDir, sprintf('%s_Vis6_OutlierCategory_Spectra.tiff', P.datePrefix));
figFile6 = fullfile(figuresDir, sprintf('%s_Vis6_OutlierCategory_Spectra.fig', P.datePrefix));
if isvalid(fig6) && isgraphics(fig6,'figure')
    fprintf('fig6 is valid before attempting to save.\n');
    try, exportgraphics(fig6, tiffFile6, 'Resolution', 300); fprintf('Plot 6 TIFF saved successfully using exportgraphics.\n');
    catch ME_export, fprintf('WARNING: exportgraphics failed for Plot 6 TIFF: %s\nAttempting print command...\n', ME_export.message);
        if isvalid(fig6) && isgraphics(fig6,'figure')
            try, print(fig6, tiffFile6, '-dtiff', '-r300'); fprintf('Plot 6 TIFF saved successfully using print command.\n');
            catch ME_print, fprintf('ERROR: print command also failed for Plot 6 TIFF: %s\n', ME_print.message); end
        else, warning('Plot 6 (fig6) became invalid before print could be attempted.');end
    end
    if isvalid(fig6) && isgraphics(fig6,'figure')
        fprintf('fig6 is still valid before savefig.\n');
        try, savefig(fig6, figFile6); fprintf('Plot 6 FIG saved successfully.\n');
        catch ME_savefig, fprintf('ERROR saving Plot 6 FIG: %s\n', ME_savefig.message);end
    else, warning('Plot 6 (fig6) became invalid before savefig for .fig file.');end
else, warning('Plot 6 figure handle (fig6) was invalid BEFORE attempting to save.');end
fprintf('All requested diagnostic visualizations generated.\n');

 % --- PLOT 7: Patient-wise Outlier Spectra Overview (Tiled Layout) ---
    % Based on 'is_OR_outlier' (local variable mapped from is_OR_outlier_all)
    % to select patients for display.
    fprintf('\nVisualization Function: Generating Plot 7: Patient-wise Outlier Spectra Overview (OR Outliers)...\n');
    
    unique_pids_plot7 = unique(Patient_ID); % Patient_ID is a local variable in this function
    patients_to_display_info_p7 = {};

    for i_pat = 1:length(unique_pids_plot7)
        current_pid_val = unique_pids_plot7{i_pat};
        patient_global_indices = find(strcmp(Patient_ID, current_pid_val));
        if isempty(patient_global_indices), continue; end
        
        % Use is_OR_outlier (local variable)
        patient_has_any_flagged_spectrum = any(is_OR_outlier(patient_global_indices));
        
        if patient_has_any_flagged_spectrum
            num_OR_flagged_spectra_for_patient = sum(is_OR_outlier(patient_global_indices));
            patient_who_grade_val = y_cat(patient_global_indices(1)); % y_cat is a local variable
            patients_to_display_info_p7{end+1,1} = {current_pid_val, num_OR_flagged_spectra_for_patient, patient_who_grade_val};
        end
    end
    
    num_patients_to_plot_p7 = size(patients_to_display_info_p7, 1);

    if num_patients_to_plot_p7 > 0
        fprintf('Plot 7: %d patients with at least one T2 or Q flagged spectrum will be displayed.\n', num_patients_to_plot_p7);
        
        ncols_layout_p7 = 6; 
        nrows_layout_p7 = ceil(num_patients_to_plot_p7 / ncols_layout_p7);
        if nrows_layout_p7 == 0, nrows_layout_p7=1; end % Ensure at least one row
        
        fig7_handle = figure('Name', 'Patienten-Ausreißer Übersicht (Trainingsdaten - OR Outliers)');
        fig_width_p7 = min(1800, ncols_layout_p7 * 250 + 100); 
        fig_height_p7 = min(1000, nrows_layout_p7 * 180 + 150);
        fig7_handle.Position = [30, 30, fig_width_p7, fig_height_p7];
        
        tl_p7 = tiledlayout(nrows_layout_p7, ncols_layout_p7, 'TileSpacing', 'compact', 'Padding', 'compact');
        title_str_p7 = sprintf('Übersicht: Ausreißer-Spektren pro Patient (T2 oder Q) - %s', datePrefix); % datePrefix is local
        try
            title(tl_p7, title_str_p7, 'FontSize', plotFontSize+2, 'FontWeight', 'normal', 'Interpreter','none');
        catch
            title(tl_p7, 'Patient Outlier Overview (OR strategy)', 'FontSize', plotFontSize+2, 'FontWeight', 'normal');
        end

        % Use local plotting parameters
        color_outlier_highlight_p7 = colorBothOutlierFlag; % Assumes colorBothOutlierFlag is defined from P_vis
        alpha_non_outlier_p7 = 0.3;
        alpha_outlier_p7 = 0.6; 
        linewidth_non_outlier_p7 = 0.5;
        linewidth_outlier_p7 = 0.8;
        
        h_legend_items_p7 = gobjects(3,1);
        legend_item_plotted_p7 = [false, false, false]; 

        for i_plot = 1:min(num_patients_to_plot_p7, nrows_layout_p7 * ncols_layout_p7)
            current_patient_info = patients_to_display_info_p7{i_plot};
            current_pid = current_patient_info{1};
            num_flagged_in_patient = current_patient_info{2};
            patient_grade = current_patient_info{3};
            
            ax_p7 = nexttile(tl_p7);
            hold(ax_p7, 'on'); box(ax_p7, 'on');

            patient_global_indices = find(strcmp(Patient_ID, current_pid));
            patient_all_spectra = X(patient_global_indices, :); % X is local
            patient_or_outlier_flags = is_OR_outlier(patient_global_indices);
            
            patient_non_outlier_spectra = patient_all_spectra(~patient_or_outlier_flags, :);
            patient_outlier_spectra = patient_all_spectra(patient_or_outlier_flags, :);
            
            color_normal_spectra_plot7 = ([0.6 0.6 0.6]); 
            legend_idx_normal = 0;
            if patient_grade == 'WHO-1', color_normal_spectra_plot7 = colorWHO1; legend_idx_normal = 1; % colorWHO1 from P_vis
            elseif patient_grade == 'WHO-3', color_normal_spectra_plot7 = colorWHO3; legend_idx_normal = 2; end % colorWHO3 from P_vis
            
            if ~isempty(patient_non_outlier_spectra)
                plot(ax_p7, wavenumbers_roi, patient_non_outlier_spectra', 'Color', [color_normal_spectra_plot7, alpha_non_outlier_p7], 'LineWidth', linewidth_non_outlier_p7, 'HandleVisibility','off');
                if legend_idx_normal > 0 && ~legend_item_plotted_p7(legend_idx_normal)
                    h_legend_items_p7(legend_idx_normal) = plot(ax_p7, NaN, NaN, 'Color', color_normal_spectra_plot7, 'LineWidth', 1.5);
                    legend_item_plotted_p7(legend_idx_normal) = true;
                end
            end
            
            if ~isempty(patient_outlier_spectra)
                plot(ax_p7, wavenumbers_roi, patient_outlier_spectra', 'Color', [color_outlier_highlight_p7, alpha_outlier_p7], 'LineWidth', linewidth_outlier_p7, 'HandleVisibility','off');
                if ~legend_item_plotted_p7(3)
                    h_legend_items_p7(3) = plot(ax_p7, NaN, NaN, 'Color', color_outlier_highlight_p7, 'LineWidth', 1.5);
                    legend_item_plotted_p7(3) = true;
                end
            end
            
            hold(ax_p7, 'off');
            xlim(ax_p7, plotXLim); ylim(ax_p7, 'auto'); 
            set(ax_p7, 'XTickMode', 'auto', 'YTickMode', 'auto');
            ax_p7.XAxis.FontSize = plotFontSize - 3; 
            ax_p7.YAxis.FontSize = plotFontSize - 3;
            ax_p7.XDir = 'reverse'; grid(ax_p7, 'off'); 
            
            title_txt = sprintf('%s (%s, %d Ausr.)', current_pid, char(patient_grade), num_flagged_in_patient);
            title(ax_p7, title_txt, 'FontSize', plotFontSize - 2, 'FontWeight', 'normal', 'Interpreter', 'none');
            
            current_tile_info_p7 = get(ax_p7,'Layout');
            current_col_p7 = mod(current_tile_info_p7.Tile-1, ncols_layout_p7) + 1;
            current_row_p7 = ceil(current_tile_info_p7.Tile / ncols_layout_p7);
            if current_col_p7 == 1, ylabel(ax_p7, 'Abs.', 'FontSize', plotFontSize-2); else set(ax_p7, 'YTickLabel', []); end
            if current_row_p7 == nrows_layout_p7, xlabel(ax_p7, plotXLabel, 'FontSize', plotFontSize-2); else set(ax_p7, 'XTickLabel', []); end % Use plotXLabel
        end
        
        legend_texts_final_p7 = {}; valid_handles_p7 = [];
        if legend_item_plotted_p7(1), valid_handles_p7 = [valid_handles_p7, h_legend_items_p7(1)]; legend_texts_final_p7{end+1} = 'Inlier WHO-1'; end
        if legend_item_plotted_p7(2), valid_handles_p7 = [valid_handles_p7, h_legend_items_p7(2)]; legend_texts_final_p7{end+1} = 'Inlier WHO-3'; end
        if legend_item_plotted_p7(3), valid_handles_p7 = [valid_handles_p7, h_legend_items_p7(3)]; legend_texts_final_p7{end+1} = 'Ausreißer (T2 oder Q)'; end
        if ~isempty(valid_handles_p7)
            lgd_p7 = legend(valid_handles_p7, legend_texts_final_p7, 'FontSize', plotFontSize-1, 'Orientation', 'horizontal');
            lgd_p7.Layout.Tile = 'South'; 
        end

        figName_p7_tiff = fullfile(figuresDir, sprintf('%s_VisFunc_Plot7_PatientWise_OR_OutlierOverview.tiff', datePrefix));
        figName_p7_fig  = fullfile(figuresDir, sprintf('%s_VisFunc_Plot7_PatientWise_OR_OutlierOverview.fig', datePrefix));
        exportgraphics(fig7_handle, figName_p7_tiff, 'Resolution', 300);
        savefig(fig7_handle, figName_p7_fig);
        fprintf('Visualization Function: Plot 7 (Patient-wise OR Outliers) saved.\n'); 
        close(fig7_handle);
    else
        fprintf('Visualization Function: Skipping Plot 7 as no patients with OR outliers were found.\n');
    end

    % --- PLOT 8: Comprehensive Patient-wise Spectra Overview (11x4 Grid, Consensus Outliers) ---
    % This uses is_T2_and_Q (local variable mapped from is_T2_and_Q_all) to highlight consensus outliers.
    fprintf('\nVisualization Function: Generating Plot 8: Comprehensive Patient-wise Spectra Overview (Consensus Outliers)...\n');
    
    unique_patient_ids_plot8 = unique(Patient_ID, 'stable');
    num_total_patients_in_train_p8 = length(unique_patient_ids_plot8);

    if num_total_patients_in_train_p8 > 0
        nrows_layout_p8 = 11; ncols_layout_p8 = 4;
        
        fig8_handle = figure('Name', 'Alle Trainingsproben: Spektren mit Konsensus-Ausreißern');
        fig_width_p8 = min(1800, ncols_layout_p8 * 300 + 100); 
        fig_height_p8 = min(1600, nrows_layout_p8 * 150 + 150); 
        fig8_handle.Position = [10, 10, fig_width_p8, fig_height_p8];
        
        tl_p8 = tiledlayout(nrows_layout_p8, ncols_layout_p8, 'TileSpacing', 'compact', 'Padding', 'compact');
        title_str_p8 = sprintf('Alle Trainingsproben: Spektren mit Konsensus-Ausreißern (Rot Markiert) - %s', datePrefix);
        try
            title(tl_p8, title_str_p8, 'FontSize', plotFontSize+4, 'FontWeight', 'bold', 'Interpreter','none');
        catch
            title(tl_p8, 'All Training Probes: Spectra with Consensus Outliers (Red Marked)', 'FontSize', plotFontSize+4, 'FontWeight', 'bold');
        end

        color_consensus_outlier_p8 = colorBothOutlierFlag; % From P_vis
        alpha_non_outlier_p8 = 0.3;
        alpha_outlier_p8 = 0.65; 
        linewidth_non_outlier_p8 = 0.5;
        linewidth_outlier_p8 = 0.8;
        
        h_legend_items_p8 = gobjects(3,1);
        legend_item_plotted_p8 = [false, false, false];

        for i_plot_pat = 1:num_total_patients_in_train_p8
            if i_plot_pat > (nrows_layout_p8 * ncols_layout_p8)
                fprintf('Plot 8 Warning: More patients (%d) than available tiles (%d). Plotting first %d.\n', ...
                    num_total_patients_in_train_p8, nrows_layout_p8 * ncols_layout_p8, nrows_layout_p8 * ncols_layout_p8);
                break;
            end
            
            current_pid_val = unique_patient_ids_plot8{i_plot_pat};
            ax_p8 = nexttile(tl_p8);
            hold(ax_p8, 'on'); box(ax_p8, 'on');

            patient_global_indices = find(strcmp(Patient_ID, current_pid_val));
            patient_all_spectra_this_probe = X(patient_global_indices, :);
            patient_consensus_outlier_flags = is_T2_and_Q(patient_global_indices); % Use local is_T2_and_Q
            
            patient_who_grade_val = y_cat(patient_global_indices(1));
            num_consensus_outliers_this_probe = sum(patient_consensus_outlier_flags);
            
            patient_non_outlier_spectra = patient_all_spectra_this_probe(~patient_consensus_outlier_flags, :);
            patient_consensus_outlier_spectra = patient_all_spectra_this_probe(patient_consensus_outlier_flags, :);
            
            color_normal_spectra_p8 = ([0.6 0.6 0.6]); 
            legend_idx_normal_p8 = 0;
            if patient_who_grade_val == 'WHO-1', color_normal_spectra_p8 = colorWHO1; legend_idx_normal_p8 = 1;
            elseif patient_who_grade_val == 'WHO-3', color_normal_spectra_p8 = colorWHO3; legend_idx_normal_p8 = 2; end
            
            if ~isempty(patient_non_outlier_spectra)
                plot(ax_p8, wavenumbers_roi, patient_non_outlier_spectra', 'Color', [color_normal_spectra_p8, alpha_non_outlier_p8], 'LineWidth', linewidth_non_outlier_p8, 'HandleVisibility','off');
                if legend_idx_normal_p8 > 0 && ~legend_item_plotted_p8(legend_idx_normal_p8)
                    h_legend_items_p8(legend_idx_normal_p8) = plot(ax_p8, NaN, NaN, 'Color', color_normal_spectra_p8, 'LineWidth', 1.5);
                    legend_item_plotted_p8(legend_idx_normal_p8) = true;
                end
            end
            
            if ~isempty(patient_consensus_outlier_spectra)
                plot(ax_p8, wavenumbers_roi, patient_consensus_outlier_spectra', 'Color', [color_consensus_outlier_p8, alpha_outlier_p8], 'LineWidth', linewidth_outlier_p8, 'HandleVisibility','off');
                if ~legend_item_plotted_p8(3)
                     h_legend_items_p8(3) = plot(ax_p8, NaN, NaN, 'Color', color_consensus_outlier_p8, 'LineWidth', 1.5);
                     legend_item_plotted_p8(3) = true;
                end
            end
            
            hold(ax_p8, 'off');
            xlim(ax_p8, plotXLim); ylim(ax_p8, 'auto'); 
            set(ax_p8, 'XTickMode', 'auto', 'YTickMode', 'auto');
            ax_p8.XAxis.FontSize = plotFontSize - 4; 
            ax_p8.YAxis.FontSize = plotFontSize - 4;
            ax_p8.XDir = 'reverse'; grid(ax_p8, 'off');
            
            title_txt_p8 = sprintf('%s (%s, %d Kons.-Ausr.)', current_pid_val, char(patient_who_grade_val), num_consensus_outliers_this_probe);
            title(ax_p8, title_txt_p8, 'FontSize', plotFontSize - 3, 'FontWeight', 'normal', 'Interpreter', 'none');
            
            current_tile_info_p8 = get(ax_p8,'Layout');
            current_col_p8 = mod(current_tile_info_p8.Tile-1, ncols_layout_p8) + 1;
            current_row_p8 = ceil(current_tile_info_p8.Tile / ncols_layout_p8);
            if current_col_p8 == 1, ylabel(ax_p8, 'Abs.', 'FontSize', plotFontSize-3); else set(ax_p8, 'YTickLabel',[]); end
            if current_row_p8 == nrows_layout_p8, xlabel(ax_p8, plotXLabel, 'FontSize', plotFontSize-3); else set(ax_p8, 'XTickLabel',[]); end % Use plotXLabel
        end
        
        for i_empty_tile = (num_total_patients_in_train_p8 + 1):(nrows_layout_p8 * ncols_layout_p8)
            ax_empty = nexttile(tl_p8, i_empty_tile); set(ax_empty, 'Visible', 'off'); 
        end

        legend_texts_final_p8 = {}; valid_handles_p8 = [];
        if legend_item_plotted_p8(1), valid_handles_p8 = [valid_handles_p8, h_legend_items_p8(1)]; legend_texts_final_p8{end+1} = 'Inlier WHO-1'; end
        if legend_item_plotted_p8(2), valid_handles_p8 = [valid_handles_p8, h_legend_items_p8(2)]; legend_texts_final_p8{end+1} = 'Inlier WHO-3'; end
        if legend_item_plotted_p8(3), valid_handles_p8 = [valid_handles_p8, h_legend_items_p8(3)]; legend_texts_final_p8{end+1} = 'Konsensus Ausreißer (T2&Q)'; end
        if ~isempty(valid_handles_p8)
            lgd_p8 = legend(valid_handles_p8, legend_texts_final_p8, 'FontSize', plotFontSize-1, 'Orientation', 'horizontal');
            lgd_p8.Layout.Tile = 'South'; 
        end

        figName_p8_tiff = fullfile(figuresDir, sprintf('%s_VisFunc_Plot8_AllPatientSpectra_ConsensusOutliers.tiff', datePrefix));
        figName_p8_fig  = fullfile(figuresDir, sprintf('%s_VisFunc_Plot8_AllPatientSpectra_ConsensusOutliers.fig', datePrefix));
        exportgraphics(fig8_handle, figName_p8_tiff, 'Resolution', 300);
        savefig(fig8_handle, figName_p8_fig);
        fprintf('Visualization Function: Plot 8 (All Patient Spectra with Consensus Outliers) saved.\n'); 
        close(fig8_handle);
    else
        fprintf('Visualization Function: Skipping Plot 8 as no patients/probes found in training data.\n');
    end

    fprintf('--- All Exploratory Visualizations Generated by Function ---\n');
    
end % End of generate_exploratory_outlier_visualizations function