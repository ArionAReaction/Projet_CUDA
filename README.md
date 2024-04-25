Pour tout les détails, consulter le rapport.

Les instructions du Makefile sont : make "nom_du_fichier"
On peut make plusieurs fichiers en même temps
Puis pour run : ./embossing-cu "nom_du_fichier"

Par exemple : make embossing_stream-cu sobel
puis pour run : ./embossing_stream-cu 14400x7200.jpg
ou bien on peut mettre : ./sobel 320x176.jpg
