# Site estático para GitHub Pages

## Estrutura
- `index.html`
- `styles.css`
- `app.js`
- `data/site-data.json`
- `images/`
- `files/`

## Como publicar no GitHub Pages
1. Crie um repositório novo no GitHub.
2. Envie todos os arquivos desta pasta para a raiz do repositório.
3. No GitHub, abra **Settings > Pages**.
4. Em **Build and deployment**, escolha **Deploy from a branch**.
5. Selecione a branch principal e a pasta `/root`.
6. Salve e aguarde a publicação.

## Observações
- Os formulários do site são somente leitura.
- A navegação pelos registros é feita localmente no navegador, sem backend.
- Para funcionamento correto, acesse via servidor HTTP/GitHub Pages, não abrindo `index.html` diretamente do disco.