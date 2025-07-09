USE pca;

CREATE DATABASE IF NOT EXISTS `pca`;

DROP TABLE IF EXISTS datasets;
CREATE TABLE datasets (
    dataset_ID int NOT NULL AUTO_INCREMENT,
    dataset_name varchar(50) NOT NULL,
    dataset_data JSON,
    dataset_algorithms JSON,
    CONSTRAINT PK_dataset PRIMARY KEY (dataset_ID)
);