#Çalışan versiyon 2 açıklamalı
import numpy as np
import matplotlib.pyplot as plt

# Cricket Algorithm fonksiyonu
def cricket_algorithm(para=[25, 0.5], function_name="Sphere", max_iter=1000):
    n = para[0]  # Popülasyon büyüklüğü
    alpha = para[1]  # Adım büyüklüğünü belirleyen parametre
    betamin = 0.2  # Minimum frekans katsayısı
    d = 2  # Boyut (optimizasyon yapılacak fonksiyonun boyutu)
    pi = np.pi  # Pi sayısı
    Lb = -5 * np.ones(d)  # Alt sınırlar (her boyut için)
    Ub = 10 * np.ones(d)  # Üst sınırlar (her boyut için)
    epsilon = 1e-10  # Sıfır bölme hatasını önlemek için küçük bir değer

    Qmin = 0  # Frekans minimum değeri
    N_iter = 0  # Toplam iterasyon sayısı, 0'dan başlatılıyor

    # Her iterasyondaki minimum değeri kaydetmek için liste
    fmin_values = []

    # Başlangıç dizileri
    Q = np.zeros((n, d))  # Frekans dizisi
    v = np.zeros((n, d))  # Hız dizisi
    Sol = Lb + (Ub - Lb) * np.random.rand(n, d)  # Çözümleri rastgele başlangıç noktalarında başlat
    Fitness = np.array([Fun(Sol[i, :], function_name) for i in range(n)])  # İlk çözümlerin fitness değerlerini hesapla

    # Başlangıç en iyi çözüm
    fmin = np.min(Fitness)  # İlk çözümler arasındaki minimum fitness değeri
    best = Sol[np.argmin(Fitness), :]  # En iyi çözümü bul
    fmin_values.append(fmin)  # İlk fmin değerini kaydet

    # İteratif döngü -- Cricket Algorithm
    while N_iter < max_iter:  # Belirlenen maksimum iterasyona kadar çalıştır
        for i in range(n):  # Her birey için döngü
            N = np.random.randint(0, 120, size=(n, d))  # Rastgele bir titreşim sayısı matrisi
            T = 0.891797 * N + 40.0252  # Titreşim sıcaklık değerleri
            T = np.clip(T, 55, 180)  # T değerini [55, 180] aralığına sınırlama
            C = (5 / 9) * (T - 32)  # Sıcaklıkları Fahrenheit'ten Celsius'a çevir
            V = 20.1 * np.sqrt(273 + C) / 1000  # Sıcaklığa göre hız değerlerini hesapla
            Z = Sol[i, :] - best  # Şu anki çözüm ile en iyi çözüm arasındaki fark

            # Z'nin sıfır olduğu durumda sıfır, diğer durumda V/Z işlemi
            F = np.where(Z == 0, 0, V[i, :] / (Z + epsilon))  # Sıfır bölme hatasını önlemek için epsilon eklenmiştir
            Q[i, :] = Qmin + (F - Qmin) * np.random.rand(d)  # Frekansı rastgele bir değişkenle güncelle
            v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i, :] + V[i, :]  # Hızı güncelle
            S = Sol[i, :] + v[i, :]  # Yeni pozisyonu hız ile güncelle

            SumF = np.sum(F) / (i + 1) + 10000  # Frekansların ortalaması
            SumT = np.sum(C) / (i + 1)  # Sıcaklıkların ortalaması
            gamma = CoefCalculate(SumF, SumT)  # Soğurulma katsayısını hesapla

            scale = Ub - Lb  # Boyut aralığını belirle
            for j in range(n):  # Tüm bireyler arasında karşılaştırma yap
                if Fitness[i] < Fitness[j]:  # Eğer i. bireyin fitness değeri daha iyi ise
                    distance = np.sqrt(np.sum((Sol[i, :] - Sol[j, :]) ** 2))  # İki birey arasındaki mesafeyi hesapla
                    PS = Fitness[i] * (4 * pi * (distance ** 2))  # Güç spektrumu
                    Lp = PS + 10 * np.log10(1 / (4 * pi * (distance ** 2)))  # Güç kaybı
                    Aatm = 7.4 * ((F ** 2) * distance) / (50 * 1e-8)  # Atmosferik zayıflama
                    RLP = Lp - Aatm  # Toplam kayıp gücü
                    K = RLP * np.exp(-gamma * distance ** 2)  # Soğurulmuş güç
                    beta = K + betamin  # Beta değerini güncelle
                    tmpf = alpha * (np.random.rand(d) - 0.5) * scale  # Rastgele bir hareket ekle
                    M = Sol[i, :] * (1 - beta) + Sol[j, :] * beta + tmpf  # Yeni çözüm hesapla
                else:
                    M = best + 0.01 * np.random.randn(d)  # En iyi çözüm yakınında rastgele çözüm

            u1 = S if np.random.rand() > gamma else M  # Seçim, yeni pozisyon veya rastgele pozisyon
            u1 = simplebounds(u1, Lb, Ub)  # Alt ve üst sınırları uygula
            Fnew = Fun(u1, function_name)  # Yeni çözümün fitness değerini hesapla

            if Fnew <= Fitness[i]:  # Eğer yeni çözüm daha iyiyse
                Sol[i, :] = u1  # Çözümü güncelle
                Fitness[i] = Fnew  # Fitness değerini güncelle
                if Fnew <= fmin:  # Eğer yeni çözüm global en iyi çözümden de iyiyse
                    best = u1  # Global en iyi çözümü güncelle
                    fmin = Fnew  # Global minimum değeri güncelle

            alpha = alpha_new(alpha)  # Alpha değerini azaltarak adım boyutunu düşür

        N_iter += 1  # İterasyon sayısını artır
        fmin_values.append(fmin)  # Her iterasyonda minimum değeri kaydet

    return best, fmin, N_iter, fmin_values

# Benchmark fonksiyonları
def Fun(u, function_name="Sphere"):
    # Belirtilen fonksiyon adına göre uygun fonksiyonu çalıştır
    if function_name == "Sphere":
        return Sphere(u)
    elif function_name == "Rastrigin":
        return Rastrigin(u)
    elif function_name == "Rosenbrock":
        return Rosenbrock(u)
    elif function_name == "Ackley":
        return Ackley(u)
    elif function_name == "Griewank":
        return Griewank(u)
    else:
        raise ValueError("Geçersiz fonksiyon adı")

# Farklı benchmark fonksiyonları
def Sphere(u):
    return np.sum(u ** 2)

def Rastrigin(u):
    A = 10
    return A * len(u) + np.sum(u ** 2 - A * np.cos(2 * np.pi * u))

def Rosenbrock(u):
    return np.sum(100 * (u[1:] - u[:-1] ** 2) ** 2 + (1 - u[:-1]) ** 2)

def Ackley(u):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(u)
    sum1 = np.sum(u ** 2)
    sum2 = np.sum(np.cos(c * u))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

def Griewank(u):
    sum_part = np.sum(u ** 2) / 4000
    prod_part = np.prod(np.cos(u / np.sqrt(np.arange(1, len(u) + 1))))
    return sum_part - prod_part + 1

# Alpha güncelleme fonksiyonu
def alpha_new(alpha):
    delta = 0.97  # Azalma katsayısı
    return delta * alpha  # Alpha değerini azalt

# Alt ve üst sınırları uygula
def simplebounds(s, Lb, Ub):
    s = np.maximum(s, Lb)  # Alt sınırları kontrol et
    s = np.minimum(s, Ub)  # Üst sınırları kontrol et
    return s

# Katsayı hesaplama fonksiyonu
def CoefCalculate(F, T):
    pres = 1  # Basınç
    relh = 50  # Bağıl nem
    freq_hum = F
    temp = T + 273
    C_humid = 4.6151 - 6.8346 * ((273.15 / temp) ** 1.261)
    hum = relh * (10 ** C_humid) * pres
    tempr = temp / 293.15
    frO = pres * (24 + 4.04e4 * hum * (0.02 + hum) / (0.391 + hum))
    frN = pres * (tempr ** -0.5) * (9 + 280 * hum * np.exp(-4.17 * ((tempr ** -1/3) - 1)))
    alpha = 8.686 * freq_hum ** 2 * (1.84e-11 * (1 / pres) * np.sqrt(tempr)
                                      + (tempr ** -2.5) * (0.01275 * np.exp(-2239.1 / temp) / (frO + freq_hum ** 2 / frO)
                                                           + 0.1068 * np.exp(-3352 / temp) / (frN + freq_hum ** 2 / frN)))
    return np.round(1000 * alpha) / 1000  # Alpha değerini hesapla ve yuvarla

# Algoritmayı belirli bir fonksiyonla çalıştır
function_name = "Sphere"
best, fmin, N_iter, fmin_values = cricket_algorithm(function_name=function_name, max_iter=1000)
print(f"En İyi Çözüm ({function_name}):", best)
print(f"Minimum Fonksiyon Değeri ({function_name}):", fmin)
print("İterasyon Sayısı:", N_iter)

# Optimizasyon sürecini grafikte göster
plt.plot(fmin_values, label=f"{function_name} Fonksiyonu")
plt.xlabel("İterasyon")
plt.ylabel("Minimum Fonksiyon Değeri")
plt.title(f"Cricket Algorithm Optimizasyon Süreci ({function_name} Fonksiyonu)")
plt.legend()
plt.show()