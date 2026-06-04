# Analyse der Serombildung nach Brustoperation bei Krebspatientinnen

Dieses Projekt untersucht, welche Variablen einen Einfluss auf die postoperative Serombildung nach Brustoperationen bei Krebspatientinnen haben. Die zugrunde liegenden Daten stammen aus einer Studie des Brustzentrums des Universitätsklinikums Aachen. Im Mittelpunkt steht die binäre Zielvariable `Serom_postoperativ`, die angibt, ob nach der Operation ein Serom aufgetreten ist oder nicht.

Ziel des Projekts ist es nicht nur, ein möglichst gutes Vorhersagemodell zu entwickeln, sondern vor allem diejenigen Variablen zu identifizieren, die den größten Einfluss auf die Serombildung haben. Der Datensatz enthält dafür unterschiedliche medizinische, demografische und klinische Informationen. Dazu gehören unter anderem Merkmale wie Alter, Gewicht, BMI, Operationsdaten, Blutwerte sowie weitere tumor- und behandlungsbezogene Variablen.

Eine zentrale Herausforderung des Projekts besteht darin, dass der Datensatz viele fehlende Werte enthält. Da fehlende Werte in medizinischen Daten häufig nicht zufällig auftreten, werden sie im Projekt nicht einfach nur entfernt. Stattdessen werden geeignete Strategien verwendet, um fehlende Werte zu behandeln und ihre Information möglichst sinnvoll in die Analyse einzubeziehen. Dazu gehören unter anderem Imputationstechniken sowie Missing-Indicator-Variablen, mit denen zusätzlich gespeichert wird, ob ein Wert ursprünglich gefehlt hat. Dadurch kann das Modell nicht nur den ersetzten Wert nutzen, sondern auch die Information, dass für eine Patientin ein bestimmter Messwert nicht vorhanden war.

Das Projekt folgt einem typischen Machine-Learning-Workflow. Zunächst werden die Daten geladen und explorativ analysiert. Dabei wird untersucht, welche Variablen vorhanden sind, wie stark die Zielvariable verteilt ist, welche Spalten besonders viele fehlende Werte enthalten und welche Features grundsätzlich für die Modellierung geeignet sind. Anschließend werden die Daten vorbereitet, indem Zielvariable und erklärende Variablen getrennt, fehlende Werte behandelt und numerische sowie kategoriale Variablen für die Modellierung nutzbar gemacht werden.

Im nächsten Schritt werden verschiedene Machine-Learning-Modelle trainiert und miteinander verglichen. Dazu gehören unter anderem Logistic Regression, Linear Support Vector Classifier, Random Forest und XGBoost. Die Modelle werden mithilfe von Cross-Validation und Hyperparameter-Optimierung bewertet, um eine möglichst robuste Einschätzung der Modellperformance zu erhalten. Als erste Baseline zeigen die Modelle bereits, dass in den Daten prädiktive Informationen zur Serombildung enthalten sind. Die Baseline-Accuracy liegt je nach Modell ungefähr zwischen 0.67 und 0.75. Besonders baumbasierte Verfahren wie Random Forest und XGBoost schneiden dabei bereits vor der finalen Feature Selection vergleichsweise gut ab.

Da der Datensatz relativ klein ist und gleichzeitig viele Features enthält, besteht ein erhöhtes Risiko für Overfitting. Deshalb liegt ein besonderer Schwerpunkt des Projekts auf der Feature Selection. Ziel ist es, den ursprünglichen Feature-Space systematisch zu reduzieren und nur diejenigen Variablen beizubehalten, die möglichst stabil und aussagekräftig mit der Zielvariable zusammenhängen. Dafür werden mehrere Verfahren kombiniert, um die Auswahl nicht nur von einem einzelnen Modell oder einer einzelnen Methode abhängig zu machen.

Zunächst werden Features mit sehr geringer Varianz sowie Variablen mit wenig Informationsgehalt untersucht. Anschließend wird Mutual Information verwendet, um mögliche nichtlineare Zusammenhänge zwischen einzelnen Features und der Zielvariable zu erkennen. Darüber hinaus werden modellbasierte Feature-Importance-Verfahren eingesetzt. Dazu gehören die Feature Importance von Random Forest und XGBoost, Koeffizienten aus regulierten linearen Modellen sowie Permutation Importance. Die Grundidee besteht darin, dass Variablen besonders interessant sind, wenn sie über mehrere unterschiedliche Methoden hinweg als relevant erkannt werden.

Zusätzlich wird eine Korrelationsanalyse durchgeführt, um stark miteinander zusammenhängende Features zu identifizieren. Dadurch können redundante Variablen erkannt und gegebenenfalls aus dem Modell entfernt werden. Beispiele für stark korrelierte Variablen im Notebook sind unter anderem `BMI_Ersterhebung` und `Gewicht_Ersterhebung`, `CD163_CD68` und `VAR00004` sowie `Blut_OP_Granulo` und `Blut_OP_Lymph`. Um Multikollinearität weiter zu analysieren, wird außerdem der Variance Inflation Factor verwendet. Dadurch kann geprüft werden, ob einzelne Features sehr stark durch andere Variablen erklärt werden und deshalb möglicherweise redundant sind.

Zur besseren Interpretierbarkeit der Modelle werden außerdem SHAP-Werte eingesetzt. SHAP ermöglicht es, den Beitrag einzelner Features zu den Modellvorhersagen genauer zu untersuchen. Dadurch kann nachvollzogen werden, welche Variablen das Modell besonders stark beeinflussen und in welche Richtung sich bestimmte Feature-Ausprägungen auf die Vorhersage auswirken. Gerade bei medizinischen Fragestellungen ist diese Interpretierbarkeit besonders wichtig, da nicht nur die reine Vorhersageleistung zählt, sondern auch die medizinische Plausibilität der gefundenen Einflussfaktoren.

Nach Kombination der verschiedenen Feature-Selection-Ansätze wird der ursprüngliche Feature-Space stark reduziert. Im Notebook werden schließlich zehn besonders relevante Features identifiziert:

`OP_Gewicht_Präp`, `Horm_KC_Dauer`, `Alter_OP`, `Blut_OP_Granulo`, `BMI_Ersterhebung`, `CD163_AT`, `OP_Revision_missing_na`, `PR_post_OP_IRS`, `VAR00004` und `OP_ax_Eingriff`.

Diese Features zeigen im Projekt die stabilste prädiktive Relevanz für die postoperative Serombildung. Sie bilden die Grundlage für die finalen Modellvergleiche.

Nach der Feature Selection werden die Modelle erneut trainiert und mit den ursprünglichen Baseline-Ergebnissen verglichen. Dabei zeigt sich, dass die Reduktion des Feature-Space die Modellperformance deutlich verbessert. Während die Baseline-Modelle ungefähr im Bereich von 0.67 bis 0.75 Accuracy liegen, erreichen die Modelle nach Feature Selection Werte von etwa 0.78 bis 0.82. Besonders XGBoost, Random Forest, k-Nearest Neighbors, RBF SVC und Logistic Regression erzielen nach der Reduktion der Features gute Ergebnisse. Die besten Modelle liegen ungefähr im Bereich von 80 Prozent Accuracy.

Zusätzlich wird untersucht, ob sich der reduzierte Feature-Space mithilfe einer Principal Component Analysis weiter komprimieren lässt. Dabei zeigt sich, dass einige Modelle auch mit einer geringeren Anzahl an Hauptkomponenten eine ähnliche Performance erreichen können. Dies deutet darauf hin, dass ein Teil der relevanten Informationen in den ausgewählten Features weiter verdichtet werden kann. Allerdings bleibt die direkte Feature-Auswahl für die medizinische Interpretation besonders wertvoll, da konkrete Variablen besser interpretierbar sind als abstrakte Hauptkomponenten.

Das Projekt ist hauptsächlich in einem Jupyter Notebook umgesetzt. Die zentrale Datei ist `main.ipynb`. Die verwendeten Daten sollten im Projektverzeichnis in einem passenden Datenordner abgelegt werden, beispielsweise unter `data/Serma_Pilotstudie.xlsx`. Eine mögliche Projektstruktur ist:

```text
.
├── data/
│   └── Serma_Pilotstudie.xlsx
├── main.ipynb
├── README.md
└── requirements.txt
```

Zur Ausführung des Projekts sollte zunächst das Repository geklont und eine virtuelle Python-Umgebung erstellt werden. Anschließend können die benötigten Pakete installiert und das Notebook gestartet werden.

```bash
git clone <repository-url>
cd <repository-name>
python -m venv venv
```

Unter Windows kann die virtuelle Umgebung mit folgendem Befehl aktiviert werden:

```bash
venv\Scripts\activate
```

Unter macOS oder Linux wird sie folgendermaßen aktiviert:

```bash
source venv/bin/activate
```

Die benötigten Abhängigkeiten können anschließend installiert werden:

```bash
pip install -r requirements.txt
```

Falls noch keine `requirements.txt` vorhanden ist, kann sie aus der aktuellen Umgebung heraus erzeugt werden:

```bash
pip freeze > requirements.txt
```

Das Notebook kann danach mit Jupyter Notebook oder JupyterLab gestartet werden:

```bash
jupyter notebook
```

oder:

```bash
jupyter lab
```

Verwendete Technologien und Bibliotheken sind unter anderem Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, SHAP, statsmodels und openpyxl.

Die Ergebnisse des Projekts zeigen, dass trotz vieler fehlender Werte und einer vergleichsweise kleinen Stichprobe relevante Signale in den Daten vorhanden sind. Durch die Kombination aus Datenvorverarbeitung, Missing-Value-Behandlung, Feature Selection, Modellvergleich und Interpretierbarkeitsmethoden konnten mehrere Variablen identifiziert werden, die mit der postoperativen Serombildung zusammenhängen. Gleichzeitig verbessert sich die Modellperformance nach der Feature Selection deutlich, was darauf hinweist, dass eine gezielte Reduktion des Feature-Space in diesem Datensatz sinnvoll ist.

Die Ergebnisse sollten dennoch vorsichtig interpretiert werden. Der Datensatz ist relativ klein, enthält viele fehlende Werte und weist ein hohes Verhältnis von Features zu Beobachtungen auf. Dadurch besteht grundsätzlich die Gefahr von Overfitting. Außerdem wurde keine externe Validierung auf einem unabhängigen Datensatz durchgeführt. Die gefundenen Variablen sollten deshalb nicht unmittelbar als klinische Entscheidungsgrundlage verstanden werden, sondern als explorative Hinweise für mögliche Einflussfaktoren, die in weiteren medizinischen und statistischen Analysen untersucht werden können.

Aus Datenschutzgründen ist bei der Veröffentlichung des Repositories besondere Vorsicht geboten. Da es sich um medizinische Studiendaten handelt, sollten sensible oder personenbezogene Informationen nicht öffentlich zugänglich gemacht werden. Falls das Repository öffentlich veröffentlicht wird, sollte geprüft werden, ob die Datei mit den Rohdaten enthalten sein darf. Andernfalls sollte der Datenordner über `.gitignore` ausgeschlossen werden, beispielsweise mit:

```text
data/
*.xlsx
```

Zusammenfassend zeigt das Projekt, wie Machine-Learning-Methoden zur Analyse medizinischer Studiendaten eingesetzt werden können, um relevante Einflussfaktoren für postoperative Serombildung zu identifizieren. Besonders wichtig sind dabei eine sorgfältige Behandlung fehlender Werte, eine robuste Feature Selection und eine interpretierbare Auswertung der Modellentscheidungen. Die finale Auswahl relevanter Variablen bietet eine kompakte Grundlage für weitere Analysen und kann als Ausgangspunkt für zukünftige medizinische Untersuchungen dienen.

