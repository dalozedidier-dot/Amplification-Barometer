Ces fixtures sont incluses uniquement pour exécuter les workflows CI "real data" de façon déterministe.

- univariate/ : séries univariées [timestamp,value] ou proches (NAB, taxis, twitter, EC2, CPC results).
- aiops_phase2/ : sous-échantillons Phase2 (2 KPI) avec colonnes timestamp,value,label,KPI ID.

Les jeux complets Phase2 (phase2_train.csv et phase2_ground_truth.hdf) sont volumineux et ne doivent pas être commités.
