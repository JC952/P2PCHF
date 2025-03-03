# 【IEEE IoTJ 2024】P2PCHF code

---

**This is the source code for "Heterogeneous Federated Learning: Client-side Collaborative Update Inter-Domain Generalization Method for Intelligent Fault Diagnosis". You can refer to the following steps to reproduce the multi-client inter-domain generalization of the fault diagnosis task.**

# Highlights

----

**A peer-to-peer communication-based heterogeneous federated learning framework is proposed for intelligent fault diagnosis, addressing model and data statistical heterogeneity among clients without a central server, enabling the construction of personalized private models with robust generalization..**

**During the collaborative model updating phase, this framework facilitates peer-to-peer interaction among heterogeneous models using an unlabeled dataset, and proposes instance and cluster dimension alignment for feature-level and semantic-level knowledge exchange, enhancing inter-domain generalization.** 

**In the local update phase, joint knowledge distillation incorporating class labels and relationships mitigates the forgetting effect and effectively balances multi-domain category knowledge.**

# Abstract

----

**The federated fault diagnosis approach has achieved remarkable results in recent years, which enables multiple clients with similar mechanical devices to collaboratively construct global intelligent diagnostic models while protecting data privacy. However, in practice, the statistical heterogeneity of data collected from different clients, as well as the model heterogeneity due to local model personalization, pose great challenges to federated learning. Meanwhile, using a central server as an information management center to build global models increases additional model parameters and the risk of data privacy leakage. To address these issues, this paper proposes a heterogeneous federated learning framework based on peer-to-peer communication (P2PCHF) for rotating machinery fault diagnosis. To achieve heterogeneous client communication without relying on a central server, sharing unlabeled dataset is utilized in the collaborative updating phase to achieve peer-to-peer communication between clients and to align instance dimensions and clustering dimensions between heterogeneous clients by constructing inter-correlation matrices to achieve feature-level and semantic-level knowledge exchange for better inter-domain generalization capabilities. Joint knowledge distillation based on class labels and class relations is introduced in the local update phase to mitigate forgetting effect in the local update phase of private models and effectively balance multi-domain category knowledge. It is verified in three cases that the proposed P2PCHF can effectively address model heterogeneity and data statistics heterogeneity among clients, and enable locally-privatized models to gain inter-domain generalization capability.**

# Proposed Method

---

![P2PFL](https://github.com/user-attachments/assets/36d90d93-6465-4820-9c48-b016d5a5648e)

![Coll](https://github.com/user-attachments/assets/d89d83fa-c7c5-43bd-a1b4-e36b02f6ca2e)

![local](https://github.com/user-attachments/assets/7b2cda18-e034-4c20-822e-a23becde019b)

# Citation
If you find this paper and repository useful, please cite us! ⭐⭐⭐
----

```
@article{ma2024heterogeneous,
  title={Heterogeneous Federated Learning: Client-side Collaborative Update Inter-Domain Generalization Method for Intelligent Fault Diagnosis},
  author={Ma, Hongbo and Wei, Jiacheng and Zhang, Guowei and Wang, Qibin and Kong, Xianguang and Du, Jingli},
  journal={IEEE Internet of Things Journal},
  year={2024},
  publisher={IEEE}
}
```

Ma H, Wei J, Zhang G, et al. Heterogeneous Federated Learning: Client-side Collaborative Update Inter-Domain Generalization Method for Intelligent Fault Diagnosis[J]. IEEE Internet of Things Journal, 2024.
# Acknowledgement

Huang W, Ye M, Shi Z, et al. Generalizable heterogeneous federated cross-correlation and instance similarity learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, 46(2): 712-728.

Huang W, Ye M, Shi Z, et al. Generalizable heterogeneous federated cross-correlation and instance similarity learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, 46(2): 712-728.
