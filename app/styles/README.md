# Système de Design TEMPOCOM

## Structure

```
styles/
├── __init__.py          # Utilitaires pour charger les styles
├── global.css           # Règles de base (body, police, variables)
├── themes.css           # Système de thèmes (light, dark, corporate)
├── components.css       # Composants réutilisables
└── README.md           # Cette documentation
```

## Utilisation

### Dans une page

```python
from components.page.Page import Page

class MaPage(Page):
    title = "Ma Page"
    theme = "dark"  # ou "light", "corporate"
    
    def render(self):
        st.title("Contenu de ma page")
```

### Variables CSS disponibles

- **Couleurs** : `--primary-color`, `--secondary-color`, `--accent-color`
- **Texte** : `--text-primary`, `--text-secondary`
- **Arrière-plans** : `--background-primary`, `--background-secondary`
- **Polices** : `--font-family-primary`, `--font-family-mono`
- **Tailles** : `--font-size-xs` à `--font-size-3xl`
- **Espacements** : `--spacing-xs` à `--spacing-2xl`

### Classes utilitaires

- **Layout** : `.tempocom-grid-2`, `.tempocom-flex-center`
- **Composants** : `.tempocom-card`, `.tempocom-button-primary`
- **Espacements** : `.mb-3`, `.mt-4`, `.p-2`

### Thèmes disponibles

1. **light** (par défaut) : Thème clair avec les couleurs TEMPOCOM
2. **dark** : Thème sombre adapté
3. **corporate** : Thème professionnel bleu
