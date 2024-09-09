# Публичный репозиторий для задачи AIJ Multi-Agent AI
English [Readme](README_eng.md)
## Описание
Данный репозиторий содержит модуль с симулятором и базовым классом агентов для задачи
[AI Journey Contest 2024 › Multiagent AI](https://dsworks.ru/champ/multiagent-ai).

## Установка из исходных файлов
### Пример для Conda
```bash
conda create -n aij_multiagent_ai python=3.11
conda activate aij_multiagent_ai
git clone https://github.com/AIRI-Institute/aij_multiagent_rl.git
cd aij_multiagent_rl
pip install -e .
```
### Запустить тесты симулятора
```bash
pytest tests
```
