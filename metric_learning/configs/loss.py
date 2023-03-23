from pytorch_metric_learning import losses

losses = {losses.ArcFaceLoss: {"num_classes": 36, "embedding_size": 128},
          losses.CosFaceLoss: {"num_classes": 36, "embedding_size": 128},
          losses.SphereFaceLoss: {"num_classes": 36, "embedding_size": 128},
          losses.SubCenterArcFaceLoss: {"num_classes": 36, "embedding_size": 128, "sub_centers": 3}
          }

