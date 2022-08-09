from models.vit_transformer import ViTConfigExtended, Backbone, LitClassifier



if __name__ == '__main__':
    configuration = ViTConfigExtended()

    backbone = Backbone(model_type='vit', config=configuration)
    model = LitClassifier(backbone)
    print(model)