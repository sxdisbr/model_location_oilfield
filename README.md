# model_location_oilfield
The task is choose the best location to drill a new well. We have samples in our data about three regions: in each there are 10000 fields, where the quality of oil and the volume of its reserves were measured.

We are going to build a machine learning model that will help determine the region where drilling will bring the greatest profit. Also, we will analize the possible profits and risks with the bootstrap technique.

These are the steps to select a location:

    Deposits are being searched for in the selected region, the values of the signs are determined for each.

    We build a model and we estimate the volume of stocks.

    The we choose the deposits with the highest estimates of values. The number of deposits depends on the company's budget and the cost of developing one well.

    The profit is equal to the total profit of the selected fields.


