% run_analysis_and_visualization_batch.m
%
% Dieses Skript führt die Phase-2-Modellselektion und die
% anschliessende Visualisierung der Projektergebnisse nacheinander aus.
%
% Stellen Sie sicher, dass MATLAB im Hauptverzeichnis des Projekts ausgeführt wird.

fprintf('=== BATCH PROCESSING STARTED: %s ===\n\n', string(datetime('now')));

% --- Schritt 1: Phase 2 Modellselektion und Vergleich durchführen ---
fprintf('--- Starte run_phase2_model_selection_comparative.m ---\n');
try
    % Stelle sicher, dass alle Variablen, die dieses Skript beeinflussen könnten,
    % sauber sind oder das Skript sich selbst initialisiert (clear, clc).
    run('run_phase2_model_selection_comparative.m'); 
    fprintf('--- run_phase2_model_selection_comparative.m erfolgreich abgeschlossen. ---\n\n');
catch ME_phase2
    fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf('FEHLER während der Ausführung von run_phase2_model_selection_comparative.m:\n');
    fprintf('%s\n', ME_phase2.message);
    fprintf('Stacktrace:\n');
    for k_stack = 1:length(ME_phase2.stack)
        fprintf('  Datei: %s, Name: %s, Zeile: %d\n', ...
            ME_phase2.stack(k_stack).file, ...
            ME_phase2.stack(k_stack).name, ...
            ME_phase2.stack(k_stack).line);
    end
    fprintf('Visualisierungsskript wird aufgrund des Fehlers NICHT gestartet.\n');
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');
    return; % Beendet das Batch-Skript bei einem Fehler in Phase 2
end

% Kurze Pause, um sicherzustellen, dass Dateisystemoperationen abgeschlossen sind (optional)
pause(2); 

% --- Schritt 2: Projektergebnisse visualisieren ---
fprintf('--- Starte run_visualize_project_results.m ---\n');
try
    % Stelle sicher, dass alle Variablen, die dieses Skript beeinflussen könnten,
    % sauber sind oder das Skript sich selbst initialisiert (clear, clc).
    run('plotting/run_visualize_project_results.m'); % Angenommen, es liegt im Unterordner 'plotting'
                                                     % Passe den Pfad ggf. an, wenn es im Haupt-src-Ordner liegt.
                                                     % Wenn es im Haupt-src-Ordner liegt, dann: run('run_visualize_project_results.m');
    fprintf('--- run_visualize_project_results.m erfolgreich abgeschlossen. ---\n\n');
catch ME_visualize
    fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf('FEHLER während der Ausführung von run_visualize_project_results.m:\n');
    fprintf('%s\n', ME_visualize.message);
    fprintf('Stacktrace:\n');
    for k_stack = 1:length(ME_visualize.stack)
        fprintf('  Datei: %s, Name: %s, Zeile: %d\n', ...
            ME_visualize.stack(k_stack).file, ...
            ME_visualize.stack(k_stack).name, ...
            ME_visualize.stack(k_stack).line);
    end
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');
    return; % Beendet das Batch-Skript bei einem Fehler in der Visualisierung
end

fprintf('=== BATCH PROCESSING FINISHED: %s ===\n', string(datetime('now')));