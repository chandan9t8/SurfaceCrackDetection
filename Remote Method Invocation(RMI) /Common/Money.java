package common;

import java.io.Serializable;

public class Money implements Serializable {
	
	private double amount;

	public Money(double amount) {
		this.amount = amount;
	}
	
	public double getAmount() {
        return this.amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }
    
    public void add(Money money) {
        this.amount += money.getAmount();
    }

    public void subtract(Money money) {
        this.amount -= money.getAmount();
    }
}