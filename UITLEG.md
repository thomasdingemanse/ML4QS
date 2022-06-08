# Uitleg git

Ik heb een fork (kopie) gemaakt van de repository van het vak. De inhoud van de originele repository en van onze fork is nu nog precies hetzelfde, behalve dit bestand met uitleg. Door zometeen met onze eigen fork verder te werken kunnen we wijzigingen maken in de code zonder dat dat effect heeft op de oorspronkelijke git repository.

## Bestaande wijzigingen bewaren

Als je al wijzigingen gemaakt hebt in de code moet je die eerst bewaren, anders raak je ze zometeen misschien kwijt. Dit kun je doen met:

```
git stash
```

## Wisselen naar onze versie van de code

Om te switchen van de oorspronkelijke code naar onze eigen fork van de code kun je instellen wat de `origin` remote URL is (dat is waar je je aanpassingen in de code naartoe wil sturen en vandaan wil halen door middel van de `push` en `pull` commands). Daarvoor kun je het volgende command gebruiken:

```
git remote set-url origin https://github.com/thomasdingemanse/ML4QS.git
```

Je kunt nu makkelijk checken of dit goed gegaan is:

```
git remote -v
```

Je zou alleen het volgende moeten zien, met `thomasdingemanse` in de URL in plaats van `mhoogen`:

```
origin  https://github.com/thomasdingemanse/ML4QS.git (fetch)
origin  https://github.com/thomasdingemanse/ML4QS.git (push)
```

## Laatste wijzigingen ophalen

Voordat je je eigen aanpassingen weer kunt terugzetten moet je eerst de meest recente wijzigingen ophalen van GitHub:

```
git pull
```

## Je wijzigingen toepassen op onze fork

Je kunt nu je bewaarde wijzigingen weer toepassen op de laatste versie van onze code met:

```
git stash apply
```

## Verdere wijzigingen

Vanaf nu kunnen we allemaal aan dezelfde code werken. Git is best slim, dus als je aan verschillende bestanden gewerkt hebt, of zelfs aan verschillende stukken van hetzelfde bestand, gaat dat eigenlijk altijd vanzelf goed. Als je toch overlappende (conflicterende) wijzigingen hebt gemaakt met iemand anders, krijg je een `merge conflict`, maar dat leg ik later nog wel uit. Het makkelijkst is om een beetje af te spreken wie waar aan werkt.

Als je aanpassingen hebt gemaakt die je wil bewaren kun je ze samenvoegen in een `commit`, een soort snapshot van de code. Daarvoor moet je eerst kiezen welke wijzigingen je aan de commit wil toevoegen. Je kunt een overzicht van je wijzigingen bekijken met:

```
git status
```

Hier staan wat bestanden die nog niet aan een commit zijn toegevoegd. Je kunt alles in één keer selecteren voor je volgende commit met (vergeet niet de punt aan het eind):

```
git add .
```

Daarmee verplaatsen ze van `unstaged changes` naar `staged changes`. Je kunt eventueel ook individuele bestanden één voor één toevoegen aan je commit met:

```
git add "naam van je bestand"
```

Je kunt nu je commit maken. Met `-m` geef je een korte beschrijving in een paar woorden:

```
git commit -m "Interpolation toegevoegd aan preprocessing"
```

## Push en pull

Jij hebt nu een commit met je laatste wijzigingen, maar wij kunnen er nog niet bij. Om je wijzigingen te delen check je meestal eerst even of er geen nieuwe wijzigingen van anderen zijn bijgekomen in de tussentijd:

```
git pull
```

Git voegt nu als het goed is alle wijzigingen van ons samen met die van jou en maakt er een nieuwe samengevoegde `merge` commit van. Als er geen nieuwe wijzigingen van anderen zijn krijg je alleen `Already up to date` te zien.

Je kunt nu je wijzigingen delen met de rest:

```
git push
```

Je kunt op [GitHub](https://github.com/thomasdingemanse/ML4QS) checken of het gelukt is door te zien of jouw versie van de code online staat.

Sorry voor de info dump :)

In de praktijk gebruik je vooral `git add`, `git commit`, `git pull`, en `git push`. De rest komt vanzelf wel als het een keer relevant wordt, maar met deze vier weet je genoeg om samen te kunnen werken met git.

Veel programma's hebben ingebouwde git support. In visual studio code kun je bijvoorbeeld in de zijbalk (3e tab) bestanden selecteren of deselecteren voor je volgende commit, een beschrijving typen, en pushen/pullen. Ik vind dat persoonlijk ideaal, maar kijk vooral wat je zelf handig vindt.