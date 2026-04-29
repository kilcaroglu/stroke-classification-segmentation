# Bilgisayarlı Tomografi Görüntülerinden Derin Öğrenme Tabanlı Otomatik İnme Tanı ve Lezyon Segmentasyon Sisteminin Geliştirilmesi

Bu proje, beyin BT (CT) görüntülerinden inme tespiti ve lezyon segmentasyonu amacıyla geliştirilmiş derin öğrenme tabanlı bir sistemdir. Sistem; sınıflandırma, segmentasyon ve bu iki yapının entegre edildiği bir pipeline yapısından oluşmaktadır.

---


## Yazarlar

- Muhammed Mustafa KİLCAROĞLU

- Alperen İLHAN

---

## Proje Amacı

İnme, erken teşhis edilmediğinde ciddi nörolojik hasarlara yol açabilen bir hastalıktır. Bu projede amaç:

- BT görüntülerinden **üç sınıflı inme sınıflandırması**
  - İnme yok
  - İskemik inme
  - Hemorajik inme

- İnmeli bölgelerde **lezyon segmentasyonu**
- Sınıflandırma ve segmentasyon modellerinin **entegre edilmesi**

---

## Veri Seti

Veri seti Sağlık Bakanlığı Açık Veri Portalı üzerinden alınmıştır:

https://acikveri.saglik.gov.tr/Home/DataSetDetail/1

Veri dağılımı:
- İnme olmayan: 4427
- İskemik inme: 1130
- Hemorajik inme: 1093

---

### Veri Bölünmesi

- Test seti: %20 (stratified)
- Eğitim + validation:
  - Sınıflandırma: 5-fold cross validation
  - Segmentasyon: %80 train / %20 validation

---

### Veri Ön İşleme

Sadece eğitim verisine veri artırma uygulanmıştır:

- ±10° rotation  
- %5 translation  
- 0.95–1.05 scaling  
- %50 horizontal flip  

 Test verisi hiçbir şekilde augment edilmemiştir.

Segmentasyon görevinde görüntü-mask uyumu korunarak aynı dönüşümler uygulanmıştır.

---

## Proje Yapısı

```text
classification
│
├── base_models
│   ├── densenet121
│   │   ├── external_test
│   │   ├── model_weights
│   │   ├── results
│   │   └── source_code
│   │
│   ├── efficientnet_b0
│   └── ...
│
└── ensemble
    ├── results
    └── source_code


segmentation
│
├── backbones_comparison
│   ├── unet
│   │   ├── model_weight
│   │   ├── results
│   │   └── source_code
│   │
│   ├── unet_attention
│   └── ...
│
└── loss_comparision
    ├── BCE_Loss
    │   ├── model_weight
    │   └── results
    │
    ├── Combo_Loss
    └── ...


stroke_detection_pipeline.py
```

---

## ÖNEMLİ UYARI / SORUMLULUK REDDİ
Bu proje, yapay zekâ ve derin öğrenme teknikleri kullanılarak geliştirilmiş deneysel bir araştırma projesidir. Yalnızca akademik çalışma, yöntem geliştirme ve eğitim amaçlıdır.

Bu sistem hiçbir koşul altında tıbbi teşhis, tedavi, klinik karar verme veya hasta yönetimi süreçlerinde kullanılmak üzere tasarlanmamıştır ve bu amaçlarla kullanılması kesinlikle yasaktır.

Sistem tarafından üretilen çıktılar tıbbi olarak doğrulanmış bilgi niteliği taşımaz ve klinik güvenilirliği garanti edilmemektedir. Bu çıktılar hiçbir şekilde gerçek hasta verisi üzerinde yorumlama, yönlendirme veya karar verme amacıyla kullanılmamalıdır.

Bu yazılım, ilgili sağlık otoriteleri ve düzenleyici kurumlar tarafından onaylanmış bir tıbbi cihaz değildir. Klinik kullanım için gerekli olan doğrulama, sertifikasyon ve yasal süreçlerden geçmemiştir.

Bu sistemin tıbbi amaçlarla kullanılması, çıktılarının klinik kararlara dahil edilmesi veya sağlık hizmeti süreçlerine entegre edilmesi durumunda ortaya çıkabilecek her türlü risk, zarar ve sonuç tamamen kullanıcıya aittir. Geliştiriciler bu tür kullanımlardan doğabilecek hiçbir sorumluluğu kabul etmez.

---

<table>
  <tr>
    <td align="center" valign="middle">
      <img src="assets/Dikey.png" alt="TÜBİTAK Logo" width="150"/>
    </td>
    <td valign="middle">
      <strong>Teşekkür / Destek</strong><br><br>
      Bu proje, TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı kapsamında desteklenerek gerçekleştirilmiştir.
      <br><br>
      Bu depoda yer alan tüm içerikler proje ekibine aittir ve TÜBİTAK’ın resmî görüşlerini yansıtmaz.
    </td>
  </tr>
</table>
