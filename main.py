import librosa as lr
import numpy as np
SR = 16000 # Частота дискретизации

def process_audio(aname):
  audio, _ = lr.load(aname, sr=SR) # Загружаем трек в память

  # Извлекаем коэффициенты
  afs = lr.feature.mfcc(audio, # из нашего звука
                        sr=SR, # с частотой дискретизации 16 кГц
                        n_mfcc=34, # Извлекаем 34 параметра
                        n_fft=2048) # Используем блоки в 125 мс
  # Суммируем все коэффициенты по времени
  # Отбрасываем два первых, так как они не слышны человеку и содержат шум
  afss = np.sum(afs[2:], axis=-1)

  # Нормализуем их
  afss = afss / np.max(np.abs(afss))

  return afss

def confidence(x, y):
  return np.sum((x - y)**2) # Евклидово расстояние
  # Меньше — лучше

## Загружаем несколько аудиодорожек
woman11 = process_audio("Women1.1.wav")
woman12 = process_audio("Women1.2.wav")
woman21 = process_audio("Women2.1.wav")
woman22= process_audio("Women2.2.wav")
man11 = process_audio("Millioner1.1.wav")
man12 = process_audio("Millioner1.2.wav")

## Сравниваем коэффициенты на близость
print('Насколько одинаковые 11 и 12', confidence(woman11, woman12))
print('Насколько одинаковые 21 и 22', confidence(woman21, woman22))
print('Насколько одинаковые голос мужчины', confidence(man11, man12))
print('Насколько одинаковые голос мужчины и женщины12', confidence(man11, woman12))
print('Насколько одинаковые 11 и 21', confidence(woman11, woman21))
print('Насколько одинаковые 11 и 22', confidence(woman11, woman22))
print('Насколько одинаковые 12 и 21', confidence(woman12, woman21))
print('Насколько одинаковые 12 и 22', confidence(woman12, woman22))
