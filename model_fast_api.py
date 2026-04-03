{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "59c8ac50-f344-4e48-a03e-7ed596752c96",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import joblib\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1f457cb2-92be-421b-ab1a-100efb84979e",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "app = FastAPI(\n",
    "    title=\"Predictive Pipeline API\",\n",
    "    description=\"An API that serves predictions from a trained scikit-learn pipeline using CSV or JSON input.\"\n",
    ")\n",
    "   \n",
    "\n",
    "# Load trained pipeline\n",
    "model = joblib.load(\"random_forest_model.pkl\")\n",
    "\n",
    "class CustomerData(BaseModel):\n",
    "    job: str\n",
    "    martial: str\n",
    "    educatiom: str\n",
    "    default: int\n",
    "    balance: int\n",
    "    housing: int\n",
    "    loan:int\n",
    "    contact: str\n",
    "    duration: int\n",
    "    campaign: int\n",
    "    pdays: int\n",
    "    previous: int\n",
    "    poutcome: str\n",
    "    \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7f126ab8-4c24-48d4-bbe2-20a62c4dc532",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.post(\"/predict\")\n",
    "def predict_churn(data: CustomerData):\n",
    "    df = pd.DataFrame([data.dict()])\n",
    "    prediction = model.predict(df)[0]\n",
    "    probability = model.predict_proba(df)[0][1]\n",
    "\n",
    "    return {\n",
    "        \"subsrciption_prediction\": int(prediction),\n",
    "        \"subsrciption_probability\": float(probability)\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3200acd-5da0-4370-8903-fbd344a208bd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
