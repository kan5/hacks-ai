# hacks-ai

# фичи
- 4 метода реализовано
- тональность работает на предобученной модели маленького руберта https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced
- кэширование есть минимальное
- бд в файле
- запускал через uvicorn
- фронт хранится на том же сервере, доступ через nginx
- хостится на ubuntu сервере 4гб озу, мб хватило бы и 2-3
- запрос кластеризации обрабатывается за 7 секунд
- если убрать тональность то за 4
