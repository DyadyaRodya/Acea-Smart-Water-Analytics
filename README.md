# Acea-Smart-Water-Analytics
Project for Data science exam.

## Выполнили:
Студенты:
- Волянский Юлиан, группа Б17-565
- Худоярова Анастасия, группа Б17-505
- Шарафиев Родион, группа Б17-503

## Описание работы
Жизнь человека невозможна без воды, поэтому аналитикам необходимо иметь представление о её доступности в водоёмах. В работе анализируются различные типы водных объектов: водные источники, озера, реки и водоносные горизонты.
Всего представлено 9 водоемов, то есть наборов данных. <br/>
Задача состоит в том, чтобы для каждого водоёма понять, что влияет на его водообеспеченность. Необходимо понять, как в течение года меняется его объем.<br/>
Временной интервал выбирается в зависимости от того, на каком интервале дается модель, то есть ежедневный, месячный или годовой, для каждого набора отдельно.<br/>
Желаемый результат: построение математических моделей, которые прогнозируют параметры, что указаны на картинке:
![](img/index.png)

Представлены 3 родника, 1 озеро, 1 река, 4 водоносных слоя. Для каждого набора надо построить свою модель. <br/>
Более подробно на изображении:
![](img/index1.png)

## Этапы работы
Были выделены следующие общие этапы работы:
1. Подготовка данных: с помощью интерполяции и логистической регрессии убраны NaN значения и восстановлены пропущенные; сортировка данных в хронологическом порядке; проверка,что данные распределены через одинаковые промежутки (восстанавливается и дополняется методами выше).
2. Получения трендов, сезонности всех характеристик.
3. Общий анализ и выявление общих закономерностей и особенностей.
4. Проверка прогнозируемых характеристик с помощью теста Дики-Фуллера.
5. Преобразование данных в случае необходимости с помощью логарифмирования или взятия разности значений.
6. По графикам ACF, PACF определяются возможные значения p,q. d устанавливается, исходя из того использовалась ли разность и какая. Также значение q, полученное из графика PACF можно заменить смещением, полученным в ADF тесте, что является более точным вариантом. 
7. Строятся возможные модели ARIMA или VAR. <br/>
Для ARIMA: Из полученных моделей по оценкам остатков модели (то есть чтобы автокорреляция на графиках ACF, PACF во всех точках была равна 0) выбирается наиболее подходящая. <br/>
Для VAR: Используется функция VARselect для подбора наиболее подходящего значения p.<br/>
8. Производится тестирование модели и валидация с помощью кросс валидации и доверительного интервала.

## Реализовано
### Petrigano
### Ход работы:
1. Подготовка данных: удаление NaN и дополнение пропусков с помощью интерполяции значений.
Пропущенные значения<br/>
![](img/depth_mis.png)
Дополненные:<br/>
![](img/depth_fixed.png)
Для volume:<br/>
Пропущенные<br/>
![](img/volume_mis.png)
Дополненные:<br/>
![](img/volume_fixed.png)
Для hydrometry:
Пропущенные:<br/>
![](img/hydrometry_mis.png)
Исправленные:<br/>
![](img/hydrometry_fix.png)
Так же проверка на пропущенные значения выполнена для temperature, rainfall, но для данных параметров данные являются полными.
1. Выставление в хронологическом порядке данных, проверка интервала между значениями временного ряда.
![](img/depth_int.png)
2. Декомпозиция временных рядов параметров Temperature, Depth, Rainfall, Volume, Hydrometry.
3. Определение максимумов и минимумов характеристик, общие выводы
![](img/1.png)
![](img/2.png)
![](img/3.png)
![](img/4.png)
![](img/4_1.png)
![](img/5.png)
![](img/6.png)
![](img/7.png)
![](img/8.png)
3. Ресемплирование для получения дополнительного подтвердждения, что данные собраны равномерно.
![](img/resample.png)
Видим, что никаких аномальных выбросов нет, поэтому нет смысла перегруппировывать. Так же сделано для параметров rainfall, volume, hyrometry, temperature.
4. Проверка данных на стационарность с помощью КПСС теста и теста Дики-Фуллера
Видим, что значение p-value больше 0,05, поэтому требуются преобразования.<br/>
До преобразований 
![](img/a1.png)
После преобразований
![](img/a2.png)
5. Приведение данных к стац виду с помощью удаления недельной и ежегодной сезонности, а также поднятие временного ряда к уровню нуля с помощью использования среднего значения.
![годовой](img/13.png)
![недельной](img/14.png)
5. Приведение временного ряда к нормальному виду
![](img/10.png)
5. Подбор параметров модели ARIMA с помощью BIC,AIC
![](img/15.png)
![](img/9.png)
Функции автокорриляции:
![](img/15.png)
![](img/16.png)
![](img/17.png)
6. Прогнозирование временного ряда 

![](11.png)

7. Предсказание данных

![](img/12.png)


### Lupa
1. Обработка данных<br/>
![](img/flow_mis.png)<br/>
![](img/mis.png)
1. Сортировка по дате и проверка, что данные расположены через 1 единицу
![](img/l_check.png)
2. Разделение данных для обучающей и тестовой выборки<br/>
![](img/separate.png)
3. Восстановление данных с помощью логистической регрессии<br/>
![](img/flow_pred.png)
4. Восстановление данных с помощью интерполяции<br/>
![](img/flow_pred_interp.png)
5. Проверка на стационарность
Видим, что прогнозируемые данные не подходят<br/>
![](img/flow_prep.png)
6. Приведение к нужному виду с помощью логарифма и квадратного корня.<br/>
![](img/flow_fix.png)
7. Подбор параметров для ARIMA с помощью ACF, PACF.<br/>
![](img/flow_acf_pcf.png)
8. Проверка, что остатки являются белым шумом<br/>
![](img/flow_est.png)
9. Предсказание
![](img/flow_predict.png)
10. Подтверждение качества модели
![](img/flow_proof.png)
11. Кросс валидация
![](img/flow_cross_valid.png)

### Bilancio
1. Сортировка по дате и проверка, что данные расположены через 1 единицу
![](img/balancio_int.png)
2. Подготовка данных
![](img/balancia_prep.png)
3. Проверка данных на стационарность
Выполнено для всех параметров Flow_Rate, level, Rainfall_S_Piero,Rainfall_Mangona, Rainfall_S_Agata, Rainfall_Cavallina, Rainfall_Le_Croci, Temperature_Le_Croci,  но приводить данные к нужному виду характеристику (Lake Level) используя diff не рекомендуют при построении VAR модели.<br/>
![](img/level_adf.png)
4. Приведение данных к нужному виду<br/>
![](img/level_diff.png)
![](img/flow_rate1_log.png)
5. Подбор параметра модели<br/>
![](img/VAR_p.png)
6. Создание тестового и тренировочного наборов<br/>
![](img/balanc_test.png)
7. Подготовка модели и проверка её параметров (p-value, F-static, R-square)<br/>
![](img/bal_mod_create.png)
![](img/bal_mod_param.png)
8. Предсказание<br/>
![](img/bal_fors.png)
9. Кросс-валидация<br/>
![](img/bal_check.png)
10. Аналогично подготавливаем другие модели, изменяя задержку выходных параметров относительно входных (так как входные влияют лишь с задержкой), также реализуем в последней модели накопительный эффект.<br/>
![](img/b_mod2.png)
![](img/b_mod3.png)
![](img/b_mod4.png)
11. Сравниваем на кросс-валидации. У последней модели лучшие показатели.<br/>
![](img/b_cv2.PNG)
![](img/b_cv3.PNG)
![](img/b_cv4.PNG)
12. Проведем кросс-валидацию, при этом во время подсчета будем трансформировать данные к исходному виду<br/>
![](img/b_cvNT.png)
