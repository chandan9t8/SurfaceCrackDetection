import java.rmi.Naming;
import java.rmi.registry.LocateRegistry;
import common.Account;
import common.AccountImpl;

public class RemoteAccount {
    public static void main(String[] args) {
        try {
            LocateRegistry.createRegistry(1099); 
            AccountImpl accountImpl = new AccountImpl();
            accountImpl.setName("John Raymond"); 
            accountImpl.setBalance(new Money(5000.0)); 
            Naming.rebind("rmi://in-csci-rrpc01.cs.iupui.edu/AccountService", accountImpl);
            System.out.println("Account Service is running...");
        } catch (Exception e) {
            System.err.println("Server exception: " + e.toString());
            e.printStackTrace();
        }
    }
}
