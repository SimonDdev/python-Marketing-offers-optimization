import pandas as pd
import pulp
import warnings

class RecommendationSystem:
    def __init__(self, model, scaler, offers: pd.DataFrame):
        self.model = model
        self.scaler = scaler
        self.offers = offers
        self.max_profit = self.offers['estimated_profit'].max()

    def is_offer_eligible(self, offer_item: pd.Series, client: dict) -> bool:
        # 1. възрастови ограничения
        if client['age'] < offer_item['min_age'] or client['age'] > offer_item['max_age']:
            return False
        # 2. целев пол
        if offer_item['target_gender'] != 'All' and offer_item['target_gender'] != client['gender']:
            return False
        # 3. бюджет (за single offer ползва клиентския бюджет,
        #    за campaign – сме задали client['budget'] = общия бюджет)
        if offer_item['price'] > client['budget']:
            return False
        # 4. предпочитана категория
        if client.get('preferred_category'):
            preferred = [p.strip() for p in str(client['preferred_category']).split(',')]
            if offer_item['category'] not in preferred:
                return False
        return True

    def optimize_offer_selection(self,
                                 eligible_offers: pd.DataFrame,
                                 scores: list) -> pd.Series:
        """
        Избира точно една оферта от eligible_offers чрез PuLP:
        max sum(scores[i] * x_i)
        s.t. sum(x_i) == 1, x_i binary.
        """
        n = len(eligible_offers)
        if n == 0:
            return None

        prob = pulp.LpProblem("Offer_Selection", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

        # целева функция
        prob += pulp.lpSum(scores[i] * x[i] for i in range(n)), "Total_Score"
        # точно една оферта
        prob += pulp.lpSum(x) == 1, "One_offer_constraint"

        prob.solve()

        for i in range(n):
            if pulp.value(x[i]) == 1:
                return eligible_offers.iloc[i]
        return None

    def get_recommendations(self, client: dict) -> pd.DataFrame:
        """
        За единичен клиент: филтрира, смята combined_score и избира една оферта.
        """
        eligible = self.offers[self.offers.apply(lambda r: self.is_offer_eligible(r, client), axis=1)]
        if eligible.empty:
            return None

        scores = []
        for _, offer in eligible.iterrows():
            fv = pd.DataFrame([[
                client['age'],
                client['income'],
                client['previous_purchases'],
                offer['price'],
                0,  # days_since_purchase (нов потребител)
                0 if client['age'] < 30 else (1 if client['age'] < 50 else 2),
                0 if client['income'] < 40000 else (1 if client['income'] < 80000 else 2),
                1 if client['previous_purchases'] >= 10 else 0,
                0,  # quantity
                0   # cross_sell_count
            ]], columns=[
                'age','income','previous_purchases','price',
                'days_since_purchase','age_group','income_bracket',
                'loyal_client','quantity','cross_sell_count'
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                propensity = self.model.predict_proba(fv)[0, 1]

            normalized_price  = offer['price'] / client['budget']
            normalized_profit = offer['estimated_profit'] / self.max_profit
            combined_score = (
                0.5 * propensity +
                0.3 * normalized_profit +
                0.2 * (1 - normalized_price)
            )
            scores.append(combined_score)

        best = self.optimize_offer_selection(eligible, scores)
        return pd.DataFrame([best]) if best is not None else None

    def optimize_campaign(self,
                          clients: pd.DataFrame,
                          total_budget: float) -> pd.DataFrame:
        """
        За множество клиенти: разпределя при най-голяма сумарна combined_score
        при общ бюджет total_budget и не повече от 1 оферта на клиент.
        """
        combos = []
        for _, client in clients.iterrows():
            c = client.to_dict()
            # задаваме глобалния бюджет като client['budget'], за да мине проверката
            c['budget'] = total_budget
            for _, offer in self.offers.iterrows():
                if not self.is_offer_eligible(offer, c):
                    continue

                fv = pd.DataFrame([[
                    c['age'], c['income'], c['previous_purchases'], offer['price'],
                    0,
                    0 if c['age'] < 30 else (1 if c['age'] < 50 else 2),
                    0 if c['income'] < 40000 else (1 if c['income'] < 80000 else 2),
                    1 if c['previous_purchases'] >= 10 else 0,
                    0, 0
                ]], columns=[
                    'age','income','previous_purchases','price',
                    'days_since_purchase','age_group','income_bracket',
                    'loyal_client','quantity','cross_sell_count'
                ])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    prop = self.model.predict_proba(fv)[0,1]

                norm_price  = offer['price'] / total_budget
                norm_profit = offer['estimated_profit'] / self.max_profit
                cs = 0.5*prop + 0.3*norm_profit + 0.2*(1 - norm_price)

                combos.append({
                    'client_id':       c['client_id'],
                    'offer_id':        offer['offer_id'],
                    'offer_name':      offer['offer_name'],
                    'price':           offer['price'],
                    'category':        offer['category'],
                    'propensity':      prop,
                    'combined_score':  cs
                })

        if not combos:
            return pd.DataFrame([])

        df = pd.DataFrame(combos)
        prob = pulp.LpProblem("Campaign_Optimization", pulp.LpMaximize)
        idx = list(df.index)
        x = pulp.LpVariable.dicts('x', idx, cat='Binary')

        # целева функция
        prob += pulp.lpSum(df.at[i,'combined_score'] * x[i] for i in idx)
        # max 1 оферта на клиент
        for client_id, group in df.groupby('client_id').groups.items():
            prob += pulp.lpSum(x[i] for i in group) <= 1
        # бюджет
        prob += pulp.lpSum(df.at[i,'price'] * x[i] for i in idx) <= total_budget

        prob.solve()
        chosen = [i for i in idx if pulp.value(x[i]) == 1]
        return df.loc[chosen].reset_index(drop=True)















