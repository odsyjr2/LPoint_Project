# 롯데멤버스 빅데이터 경진대회

Machine Learning 모델 및 XAI

## 프로젝트 구조

---

### **1. 프로젝트 개요**

롯데멤버스 LPoint의 고객 재구매율, 구매 금액을 증가시키는 개인화 마케팅 전략 수립

- **사용 데이터**
    - 롯데멤버스 제공 데이터 6개
        - LPOINT_BIG_COMP_01_DEMO : 고객 정보
        - LPOINT_BIG_COMP_02_PDDE : 상품 구매 정보
        - LPOINT_BIG_COMP_03_COP_U : 제휴사 이용 정보
        - LPOINT_BIG_COMP_04_PD_CLAC : 상품 분류 정보
        - LPOINT_BIG_COMP_05_BR : 점포 정보
        - LPOINT_BIG_COMP_06_LPAY : LPAY 이용 정보
- **사용 모델**
    1. Ensemble Model
        - Random Forest
        - XGBoost
        - Light GBM
        - CatBoost
    2. Deep Neural Network (DNN)
- **XAI:** lime, shap
- **환경:** Jupyter Notebook(Local)

### **3. EDA**

![Untitled](https://github.com/odsyjr2/LPoint_Project/assets/44573776/6a331658-d54b-4d3b-a08b-8d46ea153ad8)  
결측치 확인 및 처리

![Untitled 1](https://github.com/odsyjr2/LPoint_Project/assets/44573776/85e861b3-1b55-4e1e-a4b0-bcabd4f9ea6c)  
범주형 변수에 대한 countplot

![Untitled 2](https://github.com/odsyjr2/LPoint_Project/assets/44573776/6134a77c-e4dd-4fe1-90e7-36fbb750a05e)  
수치형 변수에 대한 boxplot

![Untitled 3](https://github.com/odsyjr2/LPoint_Project/assets/44573776/839e6597-6cea-49bf-8ee7-97aad0114ae0)  
heatmap 상관 분석


### **3. 주제 선정**

- **SWOT 분석**
    - Strengths: 20-30대 고객의 80% 이상 LPoint 가입
    - Weaknesses: 사업경쟁력 약화 추세
    - Opportunities: 최근 물가 상승으로 포인트 할인쿠폰 수요 증가
    - Threats: 코로나19로 인한 내수기반 약화
- Target 설정
    - 총 구매(이용) 횟수
    - 총 구매(이용) 금액

### **2. 데이터 전처리**

- 파생변수 생성
    - 총구매횟수(금액)
    - 총이용횟수(금액)
    - 일자별 구매(이용)횟수
    - 월 평균 구매(이용)금액
    - 구매(이용)요일
    - Lpoint 이용횟수
    - 평균 Lpoint 이용 금액
- Encoding
    - 범주형 변수 Encoding(One-Hot Encoding, Label Encoding, Binary Encoding 등)
    - 구매(이용)일자 변수를 수치로 변환
    - 구매(이용)금액 변수 및 파생변수에 Scaling - Log Transformation 적용

### **3. 모델 구성 및 학습**

Grid Search 함수 구성

- Random Forest
    
    ```python
    def RF_gridSearch(x_train, x_test, y_train, y_test):
    	algorithm = RandomForestRegressor(random_state=2022)
    	algorithm = algorithm.fit(x_train, y_train)
    	
    	depth_lst = list()
    	for es in algorithm.estimators_:
    	    depth_lst.append(es.get_depth())
    	depth = np.median(depth_lst)
    	
    	algorithm = RandomForestRegressor(random_state=2022)
    	params = {'max_depth' :[depth-1, depth, depth+1], 'n_estimators': [90,100,110], 'min_samples_split' :[2,3,4]}
    	score = 'neg_mean_squared_error'
    	
    	df_grid = GridSearchCV(algorithm, param_grid = params, cv=5, scoring= score, n_jobs=-1)
    	df_grid.fit(x_train, y_train)
    	
    	best_mse = (-1) * df_grid.best_score_
    	best_rmse = np.sqrt(best_mse)
    	print('Best score: {}, Best params: {}'.format(round(best_rmse,4), df_grid.best_params_))
    	
    	estimator = df_grid.best_estimator_
    	pred = estimator.predict(x_test)
    ```
    
- Boosting (XGB, LGB, Cat)
    
    ```python
    def Boost_GridSearch(x_train, x_test, y_train, y_test, Boost):
        if Boost == 'XGB':
            algorithm = XGBRegressor(random_state=2022)
        elif Boost == 'LGB':
            algorithm = LGBMRegressor(random_state=2022)
        else:
            algorithm = CatBoostRegressor(random_state=2022, silent=True)
            
        params = {'max_depth' :[3,5,7,11], 'learning_rate' :[0.01,0.05,0.1,0.5]}
        score = 'neg_mean_squared_error'
    
        df_grid= GridSearchCV(algorithm, param_grid = params, cv=5, scoring=score, n_jobs=-1)
        df_grid.fit(x_train, y_train)
    
        best_mse = (-1) * df_grid.best_score_
        best_rmse = np.sqrt(best_mse)
        print('Best score: {}, Best params: {}'.format(round(best_rmse,4), df_grid.best_params_))
        
        estimator = df_grid.best_estimator_
        pred = estimator.predict(x_test)
    ```
    
- DNN
    
    ```python
    def DNN_model(x_train, y_train, hidden, units):
      np.random.seed(2022)
      tf.random.set_seed(2022)
      initializer = tf.keras.initializers.GlorotUniform(seed=2022)
      
      early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
      check_point = ModelCheckpoint('temp/DNN_temp.h5',monitor='val_loss',mode='min',save_best_only=True)
      
      model = Sequential()
      model.add(keras.layers.Dense(units=units, activation='relu', input_shape=(x_train.shape[1],), kernel_initializer=initializer))
      for i in range(hidden):
          model.add(keras.layers.Dense(units=units, activation='relu', kernel_initializer=initializer))
      model.add(keras.layers.Dense(units=1, activation='linear', kernel_initializer=initializer))
      model.compile(optimizer='adam', loss='mse', metrics='mae')
      model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[early_stopping,check_point], verbose=0)
    
      return model
    ```
    

- 모델 평가 지표
    - MSE, RMSE, MAE, R² score
- 모델 성능 비교
    - CatBoost 모델이 R² score 평균 0.885로 가장 우수한 성능을 보임

### **4.  XAI**

- summary plot
    
    ![Untitled 4](https://github.com/odsyjr2/LPoint_Project/assets/44573776/4710fd10-02fb-4190-b32e-e3de26b68a75)  


- Waterfall plot

    ![Untitled 5](https://github.com/odsyjr2/LPoint_Project/assets/44573776/7735f4b4-fb4c-4c3e-b2fe-6805b8976cd9)  


- Force plot
    
    ![Untitled 6](https://github.com/odsyjr2/LPoint_Project/assets/44573776/884616ca-8270-46a7-ad2a-f8498ce6c8e9)  



- 4개 데이터셋 각각에 대한 SHAP 분석
    
    → 예측에 기여하는 주요 변수 도출
    

### **4. 기대 효과 및 개인화 마케팅 전략**

- 기대 효과
    - 고객 만족도 증가 및 롯데 멤버스 사용 횟수 증가
    - 예상 구매 금액을 기반으로 한 혜택 제공으로 총 구매 금액 증가
    - 마케팅 비용 절감
- 개인화 마케팅 전략
    - 예측한 총 구매(이용) 횟수를 기반으로 한 타겟팅 마케팅
    - 거주지와 가까운 점포에 맞춤형 상품 배치
    - 월평균 구매금액을 기준으로 한 마케팅 전략 수립

### **4. 결론**

본 프로젝트에서는 롯데멤버스의 고객 구매 데이터를 분석하고, 예측 모델을 개발하여 개인화 마케팅 전략을 제안하는 것을 목표로 하였습니다.

이를 위해 다양한 데이터를 수집하고 전처리한 후, 다양한 모델을 비교한 결과, CatBoost 모델이 가장 우수한 성능을 보여 최종 모델로 선정되었습니다. 또한 SHAP를 활용한 XAI 기법을 통해 예측 모델의 주요 변수를 분석한 결과, 상품종류 및 분류, 점포, LPpint이용횟수, 평균LPoint이용금액 등이 주요 변수로 확인되었습니다.

예측 모델을 기반으로 한 개인화 마케팅 전략은 고객의 총 구매 횟수 및 금액을 예측하여 맞춤형 혜택을 제공함으로써 고객 만족도를 높이고, 롯데멤버스의 사용 빈도를 증가시키는 것을 목표로 합니다. 또한, 마케팅 비용을 효율적으로 운용하고, 예상 구매 금액보다 높은 금액을 구매하는 고객에게 추가 혜택을 제공하여 이윤을 극대화할 수 있습니다.

결론적으로, 본 프로젝트는 데이터 분석과 예측 모델링을 통해 롯데멤버스의 개인화 마케팅 전략을 효과적으로 수립할 수 있는 기반을 마련하는 데 중점을 두었으며, 이를 통해 롯데멤버스의 경쟁력을 강화하고, 고객 만족도 및 재구매율을 높이는 데 기여할 수 있을 것으로 기대됩니다.

