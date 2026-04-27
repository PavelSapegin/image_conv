# Image Convolution

### Учебный проект по реализации операции свёртки изображений на Python.

![Original](src/images/corgy0.jpg)

### Бэнчмарки
![Original](src/images/bench_results.png)
#### Запуск
```python
pytest test_benchmark.py --benchmark-columns=min,max,mean,stddev --benchmark-sort=mean --benchmark-json=result.json 
```
Для отрисовки
```
python3 visualisation.py
```

Все изображения были взяты с `unplash.com`, распространяются под свободной лицензией.

