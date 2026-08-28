from data_ingestion import WhatdogDataModule
from model import WhatdogResNet18
import lightning as L

if __name__ == "__main__":
    data_module = WhatdogDataModule(
        images_dir="../../data/Images", 
        annotations_dir="../../data/Annotation",
        batch_size=32
    )
    
    model = WhatdogResNet18(num_classes=120, learning_rate=1e-3)
    
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto", 
        devices="auto", 
        log_every_n_steps=10
    )
    
    trainer.fit(model, datamodule=data_module)