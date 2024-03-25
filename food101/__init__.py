def get_model(args): 
    # model training type
    if args.model_type == "jlogits":
        from food101.joint_model import MultimodalFoodModel
    elif args.model_type == "ensemble":
        from food101.ensemble_model import MultimodalFoodModel
    elif args.model_type == "jprobas":
        from food101.joint_model_proba import MultimodalFoodModel
    elif args.model_type == "jprobas_jlogits": 
        from food101.joint_model_proba_logits import MultimodalFoodModel
    elif args.model_type == "ogm_ge": 
        from food101.joint_model_ogm_ge import MultimodalFoodModel
    else:   
        raise NotImplementedError("Model type not implemented")

    return MultimodalFoodModel(args)